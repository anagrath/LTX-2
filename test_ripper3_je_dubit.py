"""CPU regression checks for the local long-video/reference composition."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from ltx_pipelines.dubit import DubItPipeline
from ripper3_je_dubit import ReferenceVoicePipeline, story_result


class ReferenceCompositionTests(unittest.TestCase):
    def test_story_preparation_runs_without_autograd(self):
        class Prepared(Exception):pass
        class Fake:
            def _prepare_run(inner, **kwargs):
                self.assertTrue(torch.is_inference_mode_enabled())
                raise Prepared()
        args=SimpleNamespace(chunk_frames=185,carry_frames=25,seed=42,
            image=Path('image.png'),reference_video=Path('voice.mp4'))
        with self.assertRaises(Prepared):story_result(Fake(),args,{'prompts':['speech']})

    def test_plain_native_call_matches_the_official_setup_signature(self):
        import inspect
        from ltx_pipelines.distilled import DistilledPipeline
        from ripper3_je_native import story_result as native_story
        class Prepared(Exception):pass
        class Fake:
            def _prepare_run(inner, **kwargs):
                self.assertTrue(torch.is_inference_mode_enabled())
                inspect.signature(DistilledPipeline._prepare_run).bind(inner,**kwargs)
                raise Prepared()
        args=SimpleNamespace(chunk_frames=185,carry_frames=25,seed=42,
            num_frames=985,image=Path('image.png'))
        with self.assertRaises(Prepared):native_story(Fake(),args,{'prompts':['speech']})

    def test_native_story_wires_the_actual_distilled_context_fields(self):
        from ltx_pipelines.distilled import _DistilledRunContext
        from ltx_core.types import VIDEO_SCALE_FACTORS
        import ripper3_je_native as native
        context=_DistilledRunContext(generator=torch.Generator(),noiser=object(),
            dtype=torch.bfloat16,vae_dtype=torch.bfloat16,images=[],
            video_context=torch.zeros(1),audio_context=torch.zeros(1),num_frames=185,
            tiling_config=None,stage_1_width=288,stage_1_height=512,
            chunk_config=native.ChunkConfig(chunk_pixel_frames=185,next_video_carry_frames=0))
        pipeline=SimpleNamespace(device=torch.device('cpu'),
            _prepare_run=lambda **kwargs:context,
            _video_conditionings=lambda *args:lambda chunk:[],
            _sampler_kwargs=lambda *args:{},
            stage=SimpleNamespace(video_scale_factors=VIDEO_SCALE_FACTORS),
            upsampler=object(),video_decoder=object(),audio_decoder=object())
        args=SimpleNamespace(chunk_frames=185,carry_frames=0,seed=42,
            num_frames=185,image=Path('image.png'))
        sentinel=object()
        with patch.object(native,'pipeline_output_from_chunks',return_value=sentinel) as pack:
            self.assertIs(native.story_result(pipeline,args,{'prompts':['speech']}),sentinel)
            self.assertEqual(pack.call_args.kwargs['num_frames'],185)

    def test_target_duration_is_independent_of_reference_duration(self):
        @dataclass
        class Context:num_frames:int
        pipeline=ReferenceVoicePipeline.__new__(ReferenceVoicePipeline)
        pipeline.target_frames=985
        with patch.object(DubItPipeline,'_prepare_run',return_value=Context(169)):
            self.assertEqual(pipeline._prepare_run().num_frames,985)

    def test_late_windows_receive_the_complete_nonzero_voice_reference(self):
        pipeline=ReferenceVoicePipeline.__new__(ReferenceVoicePipeline)
        latent=torch.ones(1,8,51,16)
        condition=pipeline._audio_conditionings(latent=latent,fps=24)
        early=condition(SimpleNamespace(layout=SimpleNamespace(start_pixel_frame=0)))[0]
        late=condition(SimpleNamespace(layout=SimpleNamespace(start_pixel_frame=800)))[0]
        self.assertTrue(torch.equal(early.patchified,late.patchified))
        self.assertTrue(torch.all(late.patchified==1))
        self.assertEqual(late.patchified.shape[1],51)
        self.assertLessEqual(float(late.positions.max()),0)


if __name__=='__main__':unittest.main()
