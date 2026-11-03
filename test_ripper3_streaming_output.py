"""CPU coverage for progressive export without historical GPU output retention."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import weakref

import av
import numpy as np
import torch

from ltx_core.types import Audio
from ltx_pipelines.chunks import DecodedChunk, split_decoded_chunks
from ltx_pipelines.utils.media_io import encode_video
import ripper3_je_native as native


class StreamingOutputTests(unittest.TestCase):
    def test_split_is_lazy_releases_historical_video_and_copies_audio_to_cpu(self):
        events=[]
        video_refs=[]
        audio_refs=[]
        def chunks():
            try:
                for i in range(30):
                    self.assertTrue(all(ref() is None for ref in video_refs))
                    self.assertTrue(all(ref() is None for ref in audio_refs))
                    frames=torch.full((2,8,8,3),float(i))
                    waveform=torch.full((2,8),float(i))
                    video_refs.append(weakref.ref(frames))
                    audio_refs.append(weakref.ref(waveform))
                    events.append(i)
                    yield DecodedChunk(frames,Audio(waveform=waveform,sampling_rate=48000))
                    del frames,waveform
            finally:events.append('closed')
        video,audio,rate=split_decoded_chunks(chunks())
        self.assertEqual(events,[0])  # only enough to discover track metadata
        for i in range(30):
            frames=next(video)
            self.assertEqual(events,list(range(i+1)))
            self.assertEqual(float(frames.mean()),float(i))
            del frames
        with self.assertRaises(StopIteration):next(video)
        self.assertEqual(events[-1],'closed')
        self.assertEqual(rate,48000)
        stitched=next(audio)
        self.assertEqual(stitched.waveform.device.type,'cpu')
        expected=torch.cat([torch.full((8,2),float(i)) for i in range(30)])
        torch.testing.assert_close(stitched.waveform,expected)
        self.assertTrue(all(ref() is None for ref in video_refs+audio_refs))

    def test_story_context_keeps_residency_and_inference_during_lazy_encode_and_closes_on_error(self):
        for fail_before_consumption in (False,True):
            with self.subTest(before_consumption=fail_before_consumption):
                events=[]
                @contextmanager
                def resident():
                    events.append('resident')
                    try:yield
                    finally:events.append('release')
                def decoded():
                    try:
                        for i in range(5):
                            self.assertTrue(torch.is_inference_mode_enabled())
                            self.assertIn('resident',events)
                            self.assertNotIn('release',events)
                            events.append(i)
                            yield DecodedChunk(torch.zeros(2,8,8,3),Audio(
                                waveform=torch.zeros(2,8),sampling_rate=48000))
                    finally:events.append('decoded closed')
                source=decoded()
                pipeline=SimpleNamespace(stage=SimpleNamespace(resident=resident))
                ctx=SimpleNamespace(num_frames=10,tiling_config=None)
                with patch.object(native,'_story_chunks',return_value=(source,ctx)):
                    with self.assertRaisesRegex(RuntimeError,'encode failed'):
                        with native.story_result(pipeline,None,None) as result:
                            self.assertTrue(torch.is_inference_mode_enabled())
                            if not fail_before_consumption:
                                next(result.video)
                            raise RuntimeError('encode failed')
                self.assertEqual(events[-2:],['decoded closed','release'])
                self.assertIsNone(source.gi_frame)
                self.assertFalse(torch.is_inference_mode_enabled())

    def test_progressive_output_preserves_existing_overlap_blend_and_audio_crossfade(self):
        from ltx_pipelines.chunks import Chunk, ChunkLayout, decode_chunks, pipeline_output_from_chunks
        from ltx_pipelines.chunks.operators import _split_decoded_tail
        def decoded():
            chunks=iter([
                Chunk(ChunkLayout(17,0,0,9,3),video=torch.tensor([0]),audio=torch.tensor([0])),
                Chunk(ChunkLayout(17,8,9,0,0),video=torch.tensor([1]),audio=torch.tensor([1])),
            ])
            def video_decoder(latent,*args,**kwargs):
                return iter((torch.full((17,8,8,3),float(latent[0])*0.4),))
            def audio_decoder(latent):
                return Audio(waveform=torch.full((2,34000),0.1+float(latent[0])*0.1),
                             sampling_rate=48000)
            return decode_chunks(chunks,video_decoder,audio_decoder,fps=24)
        eager=pipeline_output_from_chunks(decoded(),num_frames=25,tiling_config=None)
        expected_video=torch.cat(list(eager.video))
        video,audio,rate=split_decoded_chunks(decoded())
        actual_video=torch.cat(list(video))
        actual_audio=next(audio)
        torch.testing.assert_close(actual_video,expected_video)
        torch.testing.assert_close(actual_audio.waveform,eager.audio.waveform)
        self.assertEqual(actual_video.shape[0],25)
        self.assertEqual(rate,48000)
        torch.testing.assert_close(actual_video[14:17,0,0,0],torch.tensor([0.1,0.2,0.3]))
        # The decoder's pending overlap is a separate allocation, not a view of
        # an entire finished video/audio window.
        original=DecodedChunk(torch.zeros(17,8,8,3),Audio(
            waveform=torch.zeros(2,34000),sampling_rate=48000))
        body,tail=_split_decoded_tail(original,tail_frames=3,prefix_audio_samples=28000)
        self.assertNotEqual(tail.video.untyped_storage().data_ptr(),original.video.untyped_storage().data_ptr())
        self.assertNotEqual(tail.audio.waveform.untyped_storage().data_ptr(),original.audio.waveform.untyped_storage().data_ptr())

    def test_progressive_encoder_muxes_all_video_and_audio_in_order(self):
        events=[]
        refs=[]
        @contextmanager
        def resident():
            events.append('resident')
            try:yield
            finally:events.append('release')
        def decoded():
            try:
                for i in range(4):
                    self.assertTrue(torch.is_inference_mode_enabled())
                    # The encoder may retain its first window, but never all history.
                    self.assertLessEqual(sum(r() is not None for r in refs),2)
                    frames=torch.full((8,32,32,3),0.1+0.2*i)
                    refs.append(weakref.ref(frames))
                    events.append(i)
                    waveform=torch.full((2,16000),0.01*(i+1))
                    yield DecodedChunk(frames,Audio(waveform=waveform,sampling_rate=48000))
                    del frames,waveform
            finally:events.append('decoded closed')
        source=decoded()
        ctx=SimpleNamespace(num_frames=32,tiling_config=None)
        pipeline=SimpleNamespace(stage=SimpleNamespace(resident=resident))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'progressive.mp4'
            with patch.object(native,'_story_chunks',return_value=(source,ctx)):
                with native.story_result(pipeline,None,None) as result:
                    encode_video(video=result.video,fps=24,audio=result.audio,
                        audio_sampling_rate=result.audio_sampling_rate,output_path=str(output),
                        video_chunks_number=4)
            self.assertEqual(events,['resident',0,1,2,3,'decoded closed','release'])
            with av.open(str(output)) as container:
                frames=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            self.assertEqual(len(frames),32)
            means=[np.mean(frames[i:i+8]) for i in range(0,32,8)]
            self.assertTrue(all(a<b for a,b in zip(means,means[1:])))
            with av.open(str(output)) as container:
                stream=container.streams.audio[0]
                audio=np.concatenate([f.to_ndarray() for f in container.decode(audio=0)],axis=-1)
                self.assertEqual(stream.rate,48000)
                self.assertEqual(stream.channels,2)
                # AAC pads a partial final packet; it must still cover all 64k samples.
                self.assertGreaterEqual(audio.shape[-1],64000)
                self.assertLess(audio.shape[-1]-64000,2048)
                self.assertTrue(np.isfinite(audio).all())
        self.assertTrue(all(ref() is None for ref in refs))


if __name__=='__main__':unittest.main()
