"""Opt-in immutable CPU preparation cache; never retains GPU model tensors."""
from __future__ import annotations

from collections import OrderedDict
import threading
import time
from pathlib import Path

import torch

from ltx_core.block_streaming.source import PinnedWeightSource
from ltx_core.block_streaming.utils import alloc_buffer, carve_buffer, derive_layout, layout_nbytes
from ltx_core.loader.primitives import StateDict


def checkpoint_stamp(paths):
    paths = [paths] if isinstance(paths, (str, Path)) else paths
    result = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        result.append((str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
    return tuple(result)


def sd_ops_identity(ops):
    if ops is None:
        return None
    # repr includes callable identity inside value transforms. This cache lives
    # only inside one process; policies must remain immutable while in use.
    return (ops.name,tuple(repr(operation) for operation in ops.mapping),
            tuple(sorted(ops.allowed_keys)) if ops.allowed_keys is not None else None)


class PreparedWeightsCache:
    """Byte-bounded CPU LRU. Borrowers keep evicted storage alive until finished.

    The budget covers retained entries; active model/source leases and in-flight
    copies can temporarily outlive eviction. clear() drains copy events before
    dropping their CPU references. It does not dispose an active model borrower.
    """

    def __init__(self,max_bytes=32*1024**3):
        if max_bytes < 0:
            raise ValueError('Prepared CPU weight budget must be nonnegative')
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.prepare_s = 0.0
        self._copies = []
        self._lock = threading.RLock()

    def _reap(self):
        self._copies = [(event,owner) for event,owner in self._copies if not event.query()]

    def get_or_prepare(self,key,prepare):
        with self._lock:
            self._reap()
            if key in self.entries:
                self.hits += 1
                value,size = self.entries.pop(key)
                self.entries[key] = value,size
                return value
            self.misses += 1
            start = time.monotonic()
            value,size = prepare()
            self.prepare_s += time.monotonic()-start
            if size <= self.max_bytes:
                while self.entries and self.bytes+size > self.max_bytes:
                    _,(_,removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
                    self.evictions += 1
                self.entries[key] = value,size
                self.bytes += size
            return value

    def pinned_source(self,key,prepare):
        def build():
            source = prepare()
            if not isinstance(source,PinnedWeightSource):
                raise TypeError('Prepared weights require a pinned RAM source')
            return source,sum(block.buffer.numel()*block.buffer.element_size() for block in source._blocks.values())
        owner = self.get_or_prepare(('packed-blocks',key),build)
        # Each streaming provider owns a fresh dictionary. Its teardown must not
        # clear the cache owner's dictionary or another active source's blocks.
        return PinnedWeightSource(dict(owner._blocks))

    def retain_copy(self,event,owner):
        with self._lock:
            self._reap()
            self._copies.append((event,owner))

    def clear(self):
        with self._lock:
            for event,_ in self._copies:
                event.synchronize()
            self._copies.clear()
            self.entries.clear()
            self.bytes = 0

    def stats(self):
        with self._lock:
            self._reap()
            return {'hits':self.hits,'misses':self.misses,'evictions':self.evictions,
                    'bytes':self.bytes,'entries':len(self.entries),'prepare_wall_s':round(self.prepare_s,3),
                    'inflight_copies':len(self._copies),'max_bytes':self.max_bytes}


class PreparedStateDictLoader:
    """Prepare transformed CPU weights once, then copy fresh tensors per build.

    Unlike the generic registry this cache never hands a mutable cached mapping
    to loaders which clear consumed state-dict entries. CUDA copies use existing
    current-stream ordering and retain pinned CPU owners until copy completion.
    No extra GPU stream or persistent GPU tensor is introduced here.
    """

    def __init__(self,loader,cache,dtype,pin_memory=True,role='projection'):
        self.loader,self.cache,self.dtype = loader,cache,dtype
        self.pin_memory,self.role = pin_memory,role

    def metadata(self,path):
        return self.loader.metadata(path)

    def load(self,paths,sd_ops=None,device=None):
        target = torch.device('cpu') if device is None else torch.device(device)
        # Direct streaming non-block loads can provide tuple shards. The actual
        # safetensors loader accepts a string or list, while the generic helper
        # normally performs this normalization before invoking it.
        path_list = [str(paths)] if isinstance(paths,(str,Path)) else [str(path) for path in paths]
        key = ('state-dict',self.role,checkpoint_stamp(paths),id(self.loader),sd_ops_identity(sd_ops),
               str(self.dtype),self.pin_memory)
        def prepare():
            loaded = self.loader.load(path_list,sd_ops=sd_ops,device=torch.device('cpu'))
            tensors = {}
            for name,value in loaded.sd.items():
                castable = value.dtype in (torch.float16,torch.float32,torch.float64,torch.bfloat16)
                preserve_scalar = value.ndim == 0 and value.dtype == torch.float32
                tensors[name] = value.to(dtype=self.dtype) if castable and not preserve_scalar else value
            layout = derive_layout(tensors)
            size = layout_nbytes(layout)
            raw = alloc_buffer(size,torch.device('cpu'),self.pin_memory)
            views = carve_buffer(raw,layout)
            for name,value in tensors.items():
                views[name].copy_(value)
            return views,size
        prepared = self.cache.get_or_prepare(key,prepare)
        tensors = {}
        try:
            for name,value in prepared.items():
                tensors[name] = value.to(device=target,non_blocking=target.type=='cuda',copy=True)
        finally:
            if target.type == 'cuda':
                # Retain the pinned owner even if a later tensor copy fails.
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(target))
                self.cache.retain_copy(event,prepared)
        return StateDict(sd=tensors,device=target,size=sum(value.nbytes for value in tensors.values()),
                         dtype={value.dtype for value in tensors.values()})
