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
