"""CPU lifecycle checks for opt-in auxiliary model reuse."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.blocks.audio_decoder import AudioDecoder
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_pipelines.utils.blocks.video_upsampler import VideoUpsampler
from ltx_core.model.video_vae.transformer import DiffVAEMode


class FakeModel:
    def __init__(self):
        self.disposed=0
        self.closed=0
        self.call_options=[]
    def eval(self):return self
    def dispose(self):self.disposed+=1
    def decode_video(self,latent,*args,**kwargs):
        self.call_options.append(kwargs)
        try:
            yield latent+1
            yield latent+2
        finally:self.closed+=1
    def decode_single_frames(self,latents,generator):
        try:yield from latents
        finally:self.closed+=1


class FakeBuilder:
    def __init__(self):
        self.models=[]
        self.options=[]
        self.module_ops=()
        self.registries=[]
    def with_registry(self,registry):
        self.registries.append(registry)
        return self
    def build(self,**kwargs):
        self.options.append(kwargs)
        model=FakeModel()
        self.models.append(model)
        return model
    def with_module_ops(self,ops):
        self.module_ops=ops
        return self


def bare(cls):
    obj=cls.__new__(cls)
    obj._resident=None
    obj._device=torch.device('cpu')
    obj._dtype=torch.bfloat16
    obj._alloc_trim_strategy=AllocatorTrimStrategy.DEFER
    obj._checkpoint_path='fake'
    obj._upsampler_path='fake upsampler'
    obj._decoder_builder=FakeBuilder()
    obj._vocoder_builder=FakeBuilder()
    obj._encoder_builder=FakeBuilder()
    obj._upsampler_builder=FakeBuilder()
    obj._diffusion_vae=False
    obj._diffvae_optimization=DiffVAEMode.CHUNKED_EAGER
    return obj


class AuxiliaryResidencyTests(unittest.TestCase):
    def test_video_defaults_to_per_call_and_close_releases_early(self):
        obj=bare(VideoDecoder)
        unused=obj(torch.ones(1))
        unused.close()
        self.assertEqual(obj._decoder_builder.models,[])
        for _ in range(2):
            stream=obj(torch.ones(1))
            next(stream)
            model=obj._decoder_builder.models[-1]
            self.assertEqual(model.disposed,0)
            stream.close()
            self.assertEqual(model.closed,1)
            self.assertEqual(model.disposed,1)
        self.assertEqual(len(obj._decoder_builder.models),2)

    def test_video_resident_is_lazy_nested_and_shared_across_decode_modes(self):
        obj=bare(VideoDecoder)
        with obj.resident():
            self.assertEqual(obj._decoder_builder.models,[])
            list(obj(torch.ones(1)))
            model=obj._decoder_builder.models[0]
            with obj.resident():
                list(obj.decode_single_frames([torch.ones(1)]))
                owned=obj.decode_owned_single_frames([torch.ones(1)])
                self.assertEqual(owned.indices,[0])
                list(owned.planes)
            self.assertEqual(len(obj._decoder_builder.models),1)
            self.assertEqual(model.disposed,0)
        self.assertEqual(model.disposed,1)
        self.assertIsNone(obj._resident)

    def test_video_dtype_and_policy_cannot_change_inside_a_resident_scope(self):
        obj=bare(VideoDecoder)
        obj._diffusion_vae=True
        with patch('ltx_pipelines.utils.blocks.video_decoder.build_diffvae_mode_op') as mode_op:
            with obj.resident():
                list(obj(torch.ones(1)))
                with self.assertRaisesRegex(ValueError,'cannot change'):
                    list(obj(torch.ones(1),dtype=torch.float32))
                list(obj(torch.ones(1)))
                obj._diffvae_optimization=DiffVAEMode.CHUNKED_COMPILE
                with self.assertRaisesRegex(ValueError,'cannot change'):
                    list(obj(torch.ones(1)))
                self.assertEqual(len(obj._decoder_builder.models),1)
            with obj.resident():
                list(obj(torch.ones(1),dtype=torch.float32))
            self.assertEqual(mode_op.call_count,2)
            self.assertEqual(mode_op.call_args.args[0],DiffVAEMode.CHUNKED_COMPILE)
            self.assertIsNotNone(mode_op.call_args.args[1])
        self.assertEqual([k['dtype'] for k in obj._decoder_builder.options],
                         [torch.bfloat16,torch.float32])
        self.assertTrue(all(m.disposed==1 for m in obj._decoder_builder.models))
        self.assertTrue(all(not r._cache_models for r in obj._decoder_builder.registries))

    def test_with_builder_copy_has_independent_residency(self):
        obj=bare(VideoDecoder)
        replacement=FakeBuilder()
        with obj.resident():
            list(obj(torch.ones(1)))
            other=obj.with_builder(replacement)
            self.assertIsNone(other._resident)
            with other.resident():
                list(other(torch.ones(1)))
            self.assertEqual(replacement.models[0].disposed,1)
            self.assertEqual(obj._decoder_builder.models[0].disposed,0)
        self.assertEqual(obj._decoder_builder.models[0].disposed,1)

    def test_scope_exit_releases_partial_iterator_and_rejects_resume(self):
        obj=bare(VideoDecoder)
        with obj.resident():
            stream=obj(torch.ones(1))
            next(stream)
            model=obj._decoder_builder.models[0]
        self.assertEqual(model.disposed,1)
        with self.assertRaisesRegex(RuntimeError,'residency scope ended'):next(stream)
        self.assertEqual(model.closed,1)
        self.assertEqual(model.disposed,1)

    def test_video_exception_releases_residency(self):
        obj=bare(VideoDecoder)
        with self.assertRaisesRegex(RuntimeError,'consumer'):
            with obj.resident():
                stream=obj(torch.ones(1))
                next(stream)
                stream.close()
                raise RuntimeError('consumer')
        self.assertEqual(obj._decoder_builder.models[0].disposed,1)

    def test_video_forwards_generator_tiling_keyframes_and_destination(self):
        obj=bare(VideoDecoder)
        generator=torch.Generator()
        destination=lambda _:torch.device('cpu')
        keyframes=object()
        with obj.resident():
            list(obj(torch.ones(1),generator=generator,keyframes=keyframes,device_fn=destination))
        self.assertEqual(obj._decoder_builder.models[0].call_options,
                         [{'keyframes':keyframes,'device_fn':destination}])

    def test_audio_and_upsampler_per_call_default_and_optional_nested_residency(self):
        cases=[(AudioDecoder,'ltx_pipelines.utils.blocks.audio_decoder.vae_decode_audio',
                ['_decoder_builder','_vocoder_builder']),
               (VideoUpsampler,'ltx_pipelines.utils.blocks.video_upsampler.upsample_video',
                ['_encoder_builder','_upsampler_builder'])]
        for cls,path,builders in cases:
            with self.subTest(block=cls.__name__):
                obj=bare(cls)
                with patch(path,return_value='decoded'):
                    for _ in range(2):self.assertEqual(obj(torch.ones(1)),'decoded')
                    for builder in builders:
                        self.assertEqual(len(getattr(obj,builder).models),2)
                        self.assertTrue(all(m.disposed==1 for m in getattr(obj,builder).models))
                    with obj.resident():
                        self.assertEqual(obj(torch.ones(1)),'decoded')
                        with obj.resident():self.assertEqual(obj(torch.ones(1)),'decoded')
                        for builder in builders:
                            self.assertEqual(len(getattr(obj,builder).models),3)
                            self.assertEqual(getattr(obj,builder).models[-1].disposed,0)
                    for builder in builders:self.assertEqual(getattr(obj,builder).models[-1].disposed,1)

    def test_audio_residency_preserves_mps_fp32_vocoder_rule(self):
        obj=bare(AudioDecoder)
        obj._device=torch.device('mps')
        with patch('ltx_pipelines.utils.blocks.audio_decoder.vae_decode_audio'):
            with obj.resident():obj(torch.ones(1))
        self.assertEqual(obj._decoder_builder.options[0]['dtype'],torch.bfloat16)
        self.assertEqual(obj._vocoder_builder.options[0]['dtype'],torch.float32)

    def test_partial_pair_build_failure_releases_already_built_model(self):
        cases=[(AudioDecoder,'_vocoder_builder','_decoder_builder'),
               (VideoUpsampler,'_upsampler_builder','_encoder_builder')]
        for cls,failing,first in cases:
            for resident in (False,True):
                with self.subTest(block=cls.__name__,resident=resident):
                    from contextlib import nullcontext
                    obj=bare(cls)
                    with patch.object(getattr(obj,failing),'build',side_effect=RuntimeError('build failed')):
                        with self.assertRaisesRegex(RuntimeError,'build failed'):
                            with obj.resident() if resident else nullcontext():obj(torch.ones(1))
                    self.assertEqual(getattr(obj,first).models[0].disposed,1)
                    self.assertIsNone(obj._resident)


if __name__=='__main__':unittest.main()
