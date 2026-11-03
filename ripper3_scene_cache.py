"""Opt-in completed-scene checkpoints; does not resume an internal diffusion window."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def stable_file_digest(path, catalog):
    """Hash model content once; rehash after size/inode/mtime/ctime changes."""
    path = Path(path).resolve()
    stat = path.stat()
    stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    previous = catalog.get(str(path))
    if previous is None or previous['stamp'] != stamp:
        previous = {'stamp':stamp,'sha256':digest(path)}
        catalog[str(path)] = previous
    return previous['sha256']


def rendering_identity(repo, models, catalog):
    """Exact checkpoint bytes, renderer sources and dependency versions identify reuse."""
    repo = Path(repo)
    code = sorted([*repo.glob('ripper3_*.py'),
                   *repo.glob('packages/ltx-core/src/**/*.py'),
                   *repo.glob('packages/ltx-pipelines/src/**/*.py'),
                   *repo.glob('packages/ltx-kernels/src/**/*.py'), repo/'uv.lock'])
    versions = {}
    for name in ('torch','transformers','av','natten'):
        try:versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name] = 'not-installed'
    return {
        'models':{str(Path(p).resolve()):stable_file_digest(p,catalog) for p in sorted(models)},
        'code':{str(p.relative_to(repo)):digest(p) for p in code},
        'dependencies':versions,
        'contract':'native-scene-artifact-v1; no hidden-state continuation',
    }


class SceneArtifacts:
    """Immutable video+manifest checkpoints, checked by content digest before reuse."""

    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def key(self, args, plan):
        previous = getattr(args, 'previous_scene_key', None)
        if previous is not None and self.lookup(previous) is None:
            raise ValueError('previous_scene_key must identify an intact completed checkpoint')
        payload = {'identity':self.identity, 'image_sha256':digest(args.image),
            'prompts':plan['prompts'], 'dialogue':plan['dialogue'], 'script':plan['script'],
            'seed':args.seed,'num_frames':args.num_frames,'chunk_frames':args.chunk_frames,
            'carry_frames':args.carry_frames,'compile':args.compile,'decoder_mode':args.decoder_mode,
            'keep_auxiliary_models':args.keep_auxiliary_models,'previous_scene_key':previous,
            'geometry':[576,1024,24],'sampler':'official LTX-2.5 Euler ancestral; FP8 cast; BF16',
        }
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def lookup(self, key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid scene checkpoint key')
        folder = self.root/key
        try:
            manifest = json.loads((folder/'manifest.json').read_text())
            if manifest['key'] == key and digest(folder/'video.mp4') == manifest['video_sha256']:
                return manifest
        except (OSError, ValueError, KeyError):
            pass
        return None

    def store(self, key, output):
        output = Path(output)
        report = json.loads((output/'result.json').read_text())
        if report.get('status') != 'completed':
            raise ValueError('Only completed scene outputs can be checkpointed')
        folder = self.root/key
        if folder.exists():
            if self.lookup(key) is None:
                raise ValueError('Existing scene checkpoint is corrupt; remove it explicitly before retrying')
            return
        temporary = self.root/('.'+key+'.'+uuid.uuid4().hex)
        temporary.mkdir(mode=0o700)
        try:
            shutil.copyfile(output/'video.mp4',temporary/'video.mp4')
            manifest = {'key':key,'video_sha256':digest(temporary/'video.mp4'),'source_report':report,
                'semantics':'completed scene artifact; not diffusion or continuity state'}
            (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            temporary.rename(folder)
        finally:
            if temporary.exists():shutil.rmtree(temporary)

    def reuse(self, key, output):
        manifest = self.lookup(key)
        if manifest is None:
            return None
        start = time.monotonic()
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        if (output/'video.mp4').exists() or (output/'result.json').exists():
            raise ValueError('Choose a fresh output directory')
        shutil.copyfile(self.root/key/'video.mp4',output/'video.mp4')
        if digest(output/'video.mp4') != manifest['video_sha256']:
            (output/'video.mp4').unlink()
            raise ValueError('Scene artifact changed during copy')
        report = {'status':'completed','artifact_reuse':True,'model_generation_performed':False,
            'generation_wall_s':0.0,'scene_cache_key':key,'source_render_wall_s':manifest['source_report'].get('wall_s'),
            'wall_s':round(time.monotonic()-start,3),'video_sha256':manifest['video_sha256']}
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
