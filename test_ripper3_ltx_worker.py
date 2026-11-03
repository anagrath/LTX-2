"""CPU checks for exact-prompt caching and serial queue lifecycle."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import torch
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ripper3_ltx_worker import PromptCache, job_args, process_pending, write_json


class WorkerTests(unittest.TestCase):
    def test_prompt_cache_is_cpu_bounded_exact_and_has_independent_outputs(self):
        calls=[]
        def encoder(prompts, **options):
            calls.append(tuple(prompts))
            return [EmbeddingsProcessorOutput(torch.ones(2),torch.ones(2),torch.ones(2)) for p in prompts]
        cache=PromptCache(encoder,'cpu',max_bytes=24)
        cache(['first'])
        hit=cache(['first'])
        hit[0].video_encoding.zero_()
        torch.testing.assert_close(cache(['first'])[0].video_encoding,torch.ones(2))
        cache(['different'])
        self.assertEqual(cache.bytes,24)
        self.assertEqual(len(cache.entries),1)
        cache(['first'])
        self.assertEqual(calls,[('first',),('different',),('first',)])
        cache(['first'],enhance_first_prompt=True)
        self.assertEqual(len(calls),4)

    def test_queue_records_errors_without_retry_and_preserves_requests(self):
        config=argparse.Namespace(compile=True,decoder_mode='chunked_compile')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            job={'output_dir':'/output/new','image':'/reference.png','story_plan':'/plan.json'}
            write_json(root/'1.job.json',job)
            write_json(root/'2.job.json',{**job,'output_dir':'/output/next'})
            seen=[]
            def fail(args):
                seen.append(str(args.output_dir))
                raise RuntimeError('test failure')
            self.assertTrue(process_pending(root,config,fail))
            self.assertEqual(json.loads((root/'1.failed.json').read_text())['error'],'test failure')
            self.assertEqual(json.loads((root/'1.request.json').read_text()),job)
            self.assertTrue(process_pending(root,config,lambda args:seen.append(str(args.output_dir))))
            self.assertEqual(seen,['/output/new','/output/next'])
            self.assertTrue((root/'2.done.json').exists())
            self.assertFalse(process_pending(root,config,fail))

    def test_jobs_cannot_change_worker_model_settings_or_use_relative_paths(self):
        config=argparse.Namespace(compile=True,decoder_mode='chunked_compile')
        job={'output_dir':'/output/new','image':'/reference.png','story_plan':'/plan.json'}
        with self.assertRaises(ValueError):job_args({**job,'decoder_mode':'other'},config)
        with self.assertRaises(ValueError):job_args({**job,'image':'relative.png'},config)
        args=job_args(job,config)
        self.assertTrue(args.keep_auxiliary_models)
        self.assertEqual(args.decoder_mode,'chunked_compile')


if __name__ == '__main__':
    unittest.main()
