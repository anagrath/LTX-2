"""CPU-only lifecycle/content checks for optional prepared prompt weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from ltx_core.block_streaming import StreamingModelBuilder
from ltx_core.block_streaming.prepared_weights import PreparedStateDictLoader, PreparedWeightsCache
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.disposable import Disposable
from ltx_pipelines.utils.blocks.prompt_encoder import PromptEncoder


class TinyModel(nn.Module,Disposable):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Linear(2,2,bias=False),nn.Linear(2,2,bias=False)])
        self.scale=nn.Parameter(torch.ones(1))

    @property
    def num_blocks(self):return len(self.blocks)

    def forward(self,value):
        for block in self.blocks:value=block(value)
        return value*self.scale


class TinyConfig:
    @classmethod
    def from_metadata(cls,metadata):return TinyModel()


class CountingLoader(SafetensorsModelStateDictLoader):
    def __init__(self):super().__init__();self.loads=[]
    def load(self,path,sd_ops=None,device=None):
        self.loads.append(sd_ops.name if sd_ops is not None else None)
        return super().load(path,sd_ops,device)


class PreparedWeightTests(unittest.TestCase):
    @staticmethod
    def checkpoint(path,scale=2):
        save_file({'blocks.0.weight':torch.eye(2),'blocks.1.weight':torch.eye(2)*3,
                   'scale':torch.tensor([float(scale)])},str(path))

    @staticmethod
    def close(model):model.teardown();model.dispose()

    def test_streaming_reuses_cpu_sources_but_tears_down_independent_providers(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tiny.safetensors';self.checkpoint(path)
            loader=CountingLoader();cache=PreparedWeightsCache(4096)
            builder=StreamingModelBuilder(TinyConfig,str(path),model_loader=loader,
                blocks_attr='blocks',blocks_prefix='blocks').with_prepared_weights(cache)
            with torch.inference_mode():
                first=builder.build(device=torch.device('cpu'),dtype=torch.float32)
                torch.testing.assert_close(first(torch.ones(1,2)),torch.full((1,2),6.0))
                self.close(first)
                self.assertEqual(len(loader.loads),2)
                second=builder.build(device=torch.device('cpu'),dtype=torch.float32)
                self.assertIsNot(first._provider,second._provider)
                self.assertIsNot(first._provider._source,second._provider._source)
                torch.testing.assert_close(second(torch.ones(1,2)),torch.full((1,2),6.0))
                self.assertEqual(len(loader.loads),2)
                # Evict cache ownership while a provider still borrows its sources.
                cache.clear()
                torch.testing.assert_close(second(torch.ones(1,2)),torch.full((1,2),6.0))
                self.close(second)
            self.assertEqual(cache.bytes,0)

    def test_changed_checkpoint_and_dtype_miss_prepared_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tiny.safetensors';self.checkpoint(path)
            loader=CountingLoader();cache=PreparedWeightsCache(8192)
            builder=StreamingModelBuilder(TinyConfig,str(path),model_loader=loader,
                blocks_attr='blocks',blocks_prefix='blocks',prepared_weights=cache)
            with torch.inference_mode():
                one=builder.build(device=torch.device('cpu'),dtype=torch.float32);self.close(one)
                two=builder.build(device=torch.device('cpu'),dtype=torch.float64);self.close(two)
                self.assertEqual(len(loader.loads),4)
                self.checkpoint(path,scale=4)
                three=builder.build(device=torch.device('cpu'),dtype=torch.float32)
                torch.testing.assert_close(three(torch.ones(1,2)),torch.full((1,2),12.0))
                self.close(three)
                self.assertEqual(len(loader.loads),6)

    def test_loras_and_disk_streaming_are_rejected_when_cache_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tiny.safetensors';self.checkpoint(path)
            builder=StreamingModelBuilder(TinyConfig,str(path),blocks_attr='blocks',blocks_prefix='blocks',
                prepared_weights=PreparedWeightsCache(8192))
            with self.assertRaisesRegex(ValueError,'no LoRAs and RAM streaming'):
                builder.build(device=torch.device('cpu'),dtype=torch.float32,cpu_slots_count=1)
            with self.assertRaisesRegex(ValueError,'no LoRAs and RAM streaming'):
                builder.with_loras((object(),)).build(device=torch.device('cpu'),dtype=torch.float32)

    def test_cpu_projection_loader_returns_fresh_mappings_and_preserves_scalar_dtype(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'weights.safetensors'
            save_file({'weight':torch.arange(4,dtype=torch.float32).reshape(2,2),
                       'scale':torch.tensor(1.125,dtype=torch.float32)},str(path))
            loader=CountingLoader();cache=PreparedWeightsCache(4096)
            staged=PreparedStateDictLoader(loader,cache,torch.bfloat16,pin_memory=False)
            first=staged.load(str(path),device=torch.device('cpu'))
            self.assertEqual(first.sd['weight'].dtype,torch.bfloat16)
            self.assertEqual(first.sd['scale'].dtype,torch.float32)
            first.sd['weight'].zero_();first.sd.clear()
            second=staged.load(str(path),device=torch.device('cpu'))
            torch.testing.assert_close(second.sd['weight'],torch.arange(4,dtype=torch.bfloat16).reshape(2,2))
            self.assertEqual(len(loader.loads),1)

    def test_prepared_loader_normalizes_tuple_shards_for_actual_safetensors_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            first=Path(temp)/'one.safetensors';second=Path(temp)/'two.safetensors'
            save_file({'first':torch.ones(2)},str(first))
            save_file({'second':torch.full((2,),2.0)},str(second))
            loader=CountingLoader();cache=PreparedWeightsCache(4096)
            staged=PreparedStateDictLoader(loader,cache,torch.float32,pin_memory=False)
            # A tuple is the real streaming builder's direct non-block call form.
            one=staged.load((str(first),str(second)),device=torch.device('cpu'))
            self.assertEqual(set(one.sd),{'first','second'})
            torch.testing.assert_close(one.sd['second'],torch.full((2,),2.0))
            # A list of equivalent paths must hit that same prepared source.
            two=staged.load([first,second],device=torch.device('cpu'))
            self.assertEqual(len(loader.loads),1)
            torch.testing.assert_close(two.sd['first'],torch.ones(2))
            save_file({'second':torch.full((2,),3.0)},str(second))
            changed=staged.load((first,second),device=torch.device('cpu'))
            self.assertEqual(len(loader.loads),2)
            torch.testing.assert_close(changed.sd['second'],torch.full((2,),3.0))

    def test_budget_eviction_and_inflight_copy_owners(self):
        cache=PreparedWeightsCache(8)
        a=cache.get_or_prepare('a',lambda:(torch.ones(2),8))
        b=cache.get_or_prepare('b',lambda:(torch.zeros(2),8))
        self.assertEqual(cache.bytes,8);self.assertEqual(cache.evictions,1)
        torch.testing.assert_close(a,torch.ones(2))
        class Event:
            done=False
            def query(self):return self.done
            def synchronize(self):self.done=True
        event=Event();cache.retain_copy(event,a)
        self.assertEqual(cache.stats()['inflight_copies'],1)
        cache.clear()
        self.assertTrue(event.done);self.assertEqual(cache.stats()['inflight_copies'],0)
        self.assertEqual(cache.bytes,0)
        with self.assertRaises(ValueError):PreparedWeightsCache(-1)

    def test_failed_cuda_transfer_keeps_cpu_owner_until_copy_event_finishes(self):
        class Event:
            done=False
            def record(self,stream):pass
            def query(self):return self.done
            def synchronize(self):self.done=True
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'weights.safetensors'
            save_file({'one':torch.ones(2),'two':torch.zeros(2)},str(path))
            cache=PreparedWeightsCache(0)
            loader=PreparedStateDictLoader(CountingLoader(),cache,torch.float32,pin_memory=False)
            original=torch.Tensor.to
            copied=[]
            def to(value,*args,**kwargs):
                if kwargs.get('device')==torch.device('cuda'):
                    copied.append(value)
                    if len(copied)>1:raise RuntimeError('copy failed')
                    return value.clone()
                return original(value,*args,**kwargs)
            event=Event()
            with patch.object(torch.Tensor,'to',to),patch('torch.cuda.Event',return_value=event),patch('torch.cuda.current_stream',return_value=object()):
                with self.assertRaisesRegex(RuntimeError,'copy failed'):
                    loader.load(str(path),device=torch.device('cuda'))
            self.assertEqual(cache.stats()['inflight_copies'],1)
            cache.clear()
            self.assertTrue(event.done)

    def test_prompt_encoder_enable_requires_ram_streaming(self):
        from ltx_pipelines.utils.types import OffloadMode
        encoder=PromptEncoder.__new__(PromptEncoder)
        encoder._offload_mode=OffloadMode.NONE
        encoder._streaming_text_encoder_builder=None
        with self.assertRaisesRegex(ValueError,'CPU RAM streaming'):
            encoder.enable_prepared_weights()

    def test_prompt_encoder_api_stages_projection_and_rejects_changed_files(self):
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_pipelines.utils.types import OffloadMode
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tiny.safetensors';self.checkpoint(path)
            loader=CountingLoader()
            encoder=PromptEncoder.__new__(PromptEncoder)
            encoder._offload_mode=OffloadMode.CPU
            encoder._streaming_text_encoder_builder=StreamingModelBuilder(TinyConfig,str(path),
                blocks_attr='blocks',blocks_prefix='blocks')
            encoder._embeddings_processor_builder=SingleGPUModelBuilder(TinyConfig,str(path),model_loader=loader)
            encoder._prepared_weights=None
            encoder._prepared_source_stamps=None
            encoder._text_encoder_path=str(path)
            encoder._embeddings_paths=[str(path)]
            encoder._dtype=torch.float32
            encoder._device=torch.device('cpu')
            with patch('ltx_pipelines.utils.blocks.prompt_encoder.EmbeddingsProcessorConfigurator.with_gemma_model_path',return_value=TinyConfig):
                self.assertIs(encoder.enable_prepared_weights(4096),encoder)
                with torch.inference_mode():
                    one=encoder._build_embeddings_processor()
                    torch.testing.assert_close(one(torch.ones(1,2)),torch.full((1,2),6.0))
                    one.dispose()
                    two=encoder._build_embeddings_processor()
                    torch.testing.assert_close(two(torch.ones(1,2)),torch.full((1,2),6.0))
                    two.dispose()
                self.assertEqual(len(loader.loads),1)
                self.assertEqual(encoder.prepared_weight_stats()['hits'],1)
                with self.assertRaises(ValueError):encoder.enable_prepared_weights(8192)
                self.checkpoint(path,scale=4)
                with self.assertRaisesRegex(RuntimeError,'model files changed'):
                    encoder._check_prepared_sources()
                encoder.clear_prepared_weights()
                self.assertEqual(encoder.prepared_weight_stats()['bytes'],0)


if __name__=='__main__':unittest.main()
