"""CPU-only checks for partial prompt hits, serial prewarm and artifact checkpoints."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ripper3_ltx_worker import PromptCache, job_args, prepare_lookahead, prewarm_prompts, process_pending, write_json
from ripper3_scene_cache import SceneArtifacts, stable_file_digest


class PromptPreparationTests(unittest.TestCase):
    def test_partial_hits_preserve_order_duplicates_and_return_independent_tensors(self):
        calls=[]
        def encode(prompts,**options):
            calls.append(prompts)
            return [EmbeddingsProcessorOutput(torch.full((2,),float(len(p))),None,torch.ones(2)) for p in prompts]
        cache=PromptCache(encode,'cpu',identity='model-A')
        cache(['one','second'])
        values=cache(['second','new','one','new'])
        self.assertEqual(calls,[['one','second'],['new']])
        self.assertEqual([float(v.video_encoding[0]) for v in values],[6,3,3,3])
        values[1].video_encoding.zero_()
        self.assertEqual(float(values[3].video_encoding[0]),3)
        self.assertEqual(float(cache(['new'])[0].video_encoding[0]),3)
        self.assertTrue(all(v.video_encoding.device.type=='cpu' for v,_ in cache.entries.values()))
        cache.identity='model-B'
        cache(['one'])
        self.assertEqual(calls[-1],['one'])

    def test_prewarm_preserves_cpu_rng_and_does_not_materialize_hits(self):
        def encode(prompts,**options):
            return [EmbeddingsProcessorOutput(torch.rand(2),None,torch.ones(2)) for p in prompts]
        cache=PromptCache(encode,'cpu')
        before=torch.get_rng_state().clone()
        details=prewarm_prompts(cache,'cpu',['one','two'])
        torch.testing.assert_close(torch.get_rng_state(),before)
        self.assertEqual(details['cache_misses'],2)
        with patch.object(cache,'_copy',side_effect=AssertionError('cached prewarm must not materialize')):
            prewarm_prompts(cache,'cpu',['one','two'])
        self.assertEqual(cache.hits,2)

    def test_unknown_or_enhanced_options_do_not_hit_exact_caption_cache(self):
        calls=[]
        def encode(prompts,**options):
            calls.append(options)
            return [EmbeddingsProcessorOutput(torch.ones(2),None,torch.ones(2)) for p in prompts]
        cache=PromptCache(encode,'cpu')
        cache(['one'])
        cache(['one'],enhance_first_prompt=True,enhance_prompt_image='different.png')
        cache(['one'],new_conditioning_option='value')
        self.assertEqual(len(calls),3)
        self.assertEqual(cache.hits,0)

    def test_prewarm_queue_receipt_is_distinct_from_model_generation(self):
        config=argparse.Namespace(compile=True,decoder_mode='chunked_compile')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'one.job.json',{'action':'prewarm','story_plan':'/plan.json'})
            events=[]
            self.assertTrue(process_pending(root,config,lambda args:events.append(args.action) or {'prewarm_wall_s':1.0}))
            receipt=json.loads((root/'one.done.json').read_text())
            self.assertEqual(events,['prewarm'])
            self.assertIsNone(receipt['output_dir'])
            self.assertEqual(receipt['action'],'prewarm')
            self.assertEqual(receipt['details']['prewarm_wall_s'],1.0)

    def test_boundary_lookahead_deduplicates_and_skips_bad_future_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=root/'first.json';second=root/'second.json'
            write_json(first,{'prompts':['one','shared']})
            write_json(second,{'prompts':['shared','next']})
            for i,path in [(1,first),(3,second)]:
                write_json(root/f'{i}.job.json',{'action':'prewarm','story_plan':str(path)})
            write_json(root/'2.job.json',{'action':'prewarm','story_plan':'relative.json'})
            config=argparse.Namespace(compile=True,decoder_mode='chunked_compile',lookahead=2)
            events=[]
            prepare_lookahead(root,config,lambda prompts:events.append(('prepare',prompts)))
            process_pending(root,config,lambda args:events.append(('run',args.action)))
            self.assertEqual(events,[('prepare',['one','shared','next']),('run','prewarm')])
            self.assertTrue((root/'2.job.json').exists())

    def test_cache_eviction_is_per_caption_and_oversized_values_are_not_retained(self):
        def encode(prompts,**options):
            return [EmbeddingsProcessorOutput(torch.ones(2),None,torch.ones(2)) for p in prompts]
        cache=PromptCache(encode,'cpu',max_bytes=16)
        cache(['one','two'])
        self.assertEqual(cache.bytes,16)
        self.assertEqual([key[1] for key in cache.entries],['two'])
        tiny=PromptCache(encode,'cpu',max_bytes=1)
        tiny(['one'])
        self.assertEqual(tiny.bytes,0)
        self.assertFalse(tiny.entries)


class SceneCheckpointTests(unittest.TestCase):
    @staticmethod
    def args(root):
        image=root/'image.png';image.write_bytes(b'fake-reference-content')
        return argparse.Namespace(image=image,seed=42,num_frames=145,chunk_frames=145,carry_frames=0,
            compile=True,decoder_mode='chunked_compile',keep_auxiliary_models=True)

    def test_completed_scene_reuse_is_explicit_and_does_not_claim_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root)
            cache=SceneArtifacts(root/'cache',{'models':'exact-checkpoint-digest'})
            plan={'prompts':['first'],'dialogue':['hello'],'script':'hello'}
            key=cache.key(args,plan)
            output=root/'original';output.mkdir()
            (output/'video.mp4').write_bytes(b'completed-video')
            write_json(output/'result.json',{'status':'completed','wall_s':120.0,'peak_allocated_bytes':999})
            cache.store(key,output)
            reused=cache.reuse(key,root/'reused')
            self.assertTrue(reused['artifact_reuse'])
            self.assertFalse(reused['model_generation_performed'])
            self.assertEqual(reused['generation_wall_s'],0.0)
            self.assertEqual(reused['source_render_wall_s'],120.0)
            self.assertNotIn('peak_allocated_bytes',reused)
            self.assertEqual((root/'reused/video.mp4').read_bytes(),b'completed-video')
            with self.assertRaises(ValueError):cache.reuse(key,root/'reused')
            (root/'cache'/key/'video.mp4').write_bytes(b'corrupt')
            self.assertIsNone(cache.lookup(key))

    def test_later_scene_edits_preserve_prior_scene_key_and_invalidate_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root)
            cache=SceneArtifacts(root/'cache',{'models':'model-A'})
            first={'prompts':['scene-one'],'dialogue':['one'],'script':'one'}
            first_key=cache.key(args,first)
            output=root/'original';output.mkdir()
            (output/'video.mp4').write_bytes(b'one')
            write_json(output/'result.json',{'status':'completed','wall_s':1.0})
            cache.store(first_key,output)
            args.previous_scene_key=first_key
            later={'prompts':['scene-two'],'dialogue':['two'],'script':'two'}
            second_key=cache.key(args,later)
            changed={**later,'prompts':['changed scene-two']}
            self.assertNotEqual(second_key,cache.key(args,changed))
            del args.previous_scene_key
            self.assertEqual(first_key,cache.key(args,first))
            args.seed=43
            self.assertNotEqual(first_key,cache.key(args,first))
            args.seed=42
            args.image.write_bytes(b'changed reference')
            self.assertNotEqual(first_key,cache.key(args,first))
            self.assertNotEqual(first_key,SceneArtifacts(root/'other',{'models':'model-B'}).key(args,first))

    def test_model_content_digest_is_memoized_and_rehashes_changed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model';path.write_bytes(b'original')
            catalog={}
            first=stable_file_digest(path,catalog)
            with patch('ripper3_scene_cache.digest',side_effect=AssertionError('must reuse fingerprint')):
                self.assertEqual(stable_file_digest(path,catalog),first)
            path.write_bytes(b'modified')
            self.assertNotEqual(stable_file_digest(path,catalog),first)


if __name__=='__main__':unittest.main()
