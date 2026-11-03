"""CPU-only checks for cache-aware CUDA AUTO decode planning; no CUDA context."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ltx_core.devices import cuda_allocatable_bytes
from ltx_core.model.video_vae.diffusion_tiling import budget_safety_bytes
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_pipelines.utils import helpers


GIB = 1024**3
# Pinned LTX-2.5 decoder geometry. Remaining stage kernels/upsamples use its
# matching defaults; this creates no model and does not need local weights.
VAE_CONFIG = {"decoder": {"stage_channels": [2048, 1024, 512, 512, 256], "head_dim": 64, "patch_size": 4}}


class DecodeBudgetTests(unittest.TestCase):
    def cuda_state(self, stack, *, allocated=6*GIB, reserved=6*GIB, free=18*GIB, fraction=1.0):
        # Every CUDA entry point used by the estimator is mocked, including
        # availability. Tests still run with CUDA_VISIBLE_DEVICES empty.
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(torch.cuda, 'mem_get_info', return_value=(free, 24*GIB)))
        stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
            return_value=SimpleNamespace(total_memory=24*GIB)))
        stack.enter_context(patch.object(torch.cuda, 'get_per_process_memory_fraction', return_value=fraction))
        stack.enter_context(patch.object(torch.cuda, 'memory_allocated', return_value=allocated))
        stack.enter_context(patch.object(torch.cuda, 'memory_reserved', return_value=reserved))

    def fake_checkpoint(self, stack):
        stack.enter_context(patch.object(helpers, 'is_diffusion_video_vae', return_value=True))
        loader = stack.enter_context(patch.object(helpers, 'SafetensorsModelStateDictLoader'))
        loader.return_value.metadata.return_value = {'config': {'vae': VAE_CONFIG}}
        stack.enter_context(patch.object(helpers, 'estimate_diffusion_decoder_weight_bytes', return_value=3*GIB//2))
        # A fixed backend recipe keeps this geometry test independent of optional
        # NATTEN installation, and never asks about a real GPU's capabilities.
        stack.enter_context(patch('ltx_core.model.video_vae.diffusion_tiling._falls_back_to_eager_na', return_value=False))

    def test_warm_allocator_does_not_shrink_identical_decode_tiles(self):
        plans = []
        for reserved, free in [(6*GIB, 18*GIB), (12*GIB, 12*GIB)]:
            with ExitStack() as stack:
                self.cuda_state(stack, reserved=reserved, free=free)
                self.fake_checkpoint(stack)
                old_probe = stack.enter_context(patch.object(helpers, 'activation_budget_bytes',
                    side_effect=AssertionError('CUDA AUTO must account for reusable cache')))
                plans.append(helpers.tiling_config_for_vae('fake.safetensors',
                    height=1024, width=576, num_frames=185,
                    diffvae_optimization=DiffVAEMode.CHUNKED_COMPILE, device=torch.device('cuda:0')))
                old_probe.assert_not_called()
        self.assertEqual(plans[0], plans[1])

    def test_allocatable_budget_honors_external_usage_and_fraction_limit(self):
        with ExitStack() as stack:
            self.cuda_state(stack, reserved=10*GIB, free=7*GIB)
            # Four GiB reusable in this process + seven GiB genuinely free;
            # the rest of the device belongs to live tensors/other processes.
            self.assertEqual(cuda_allocatable_bytes(torch.device('cuda:0')), 11*GIB)
        with ExitStack() as stack:
            self.cuda_state(stack, reserved=10*GIB, free=14*GIB, fraction=0.5)
            self.assertEqual(cuda_allocatable_bytes(torch.device('cuda:0')), 6*GIB)
        with ExitStack() as stack:
            self.cuda_state(stack, allocated=13*GIB, reserved=14*GIB, free=10*GIB, fraction=0.5)
            self.assertEqual(cuda_allocatable_bytes(torch.device('cuda:0')), 0)

    def test_explicit_budget_and_weight_reserve_are_preserved(self):
        sentinel = object()
        with ExitStack() as stack:
            self.fake_checkpoint(stack)
            cuda = stack.enter_context(patch.object(helpers, 'cuda_allocatable_bytes',
                side_effect=AssertionError('explicit budget must bypass probing')))
            other = stack.enter_context(patch.object(helpers, 'activation_budget_bytes',
                side_effect=AssertionError('explicit budget must bypass probing')))
            recommend = stack.enter_context(patch.object(helpers, 'recommended_decode_tiling_config', return_value=sentinel))
            result = helpers.tiling_config_for_vae('fake.safetensors', height=1024, width=576,
                num_frames=185, device=torch.device('cuda:0'), free_bytes=8*GIB,
                diffvae_optimization=DiffVAEMode.CHUNKED_COMPILE)
            self.assertIs(result, sentinel)
            self.assertEqual(recommend.call_args.kwargs['free_bytes'], 8*GIB)
            # Still charge estimated weights, even for a worker that may already
            # hold them. Removing that conservative charge needs separate evidence.
            self.assertEqual(recommend.call_args.kwargs['model_bytes'], 3*GIB//2)
            self.assertEqual(budget_safety_bytes(DiffVAEMode.CHUNKED_COMPILE), 2*GIB)
            cuda.assert_not_called()
            other.assert_not_called()

    def test_cpu_mps_and_conv_paths_do_not_probe_cuda(self):
        for device in [torch.device('cpu'), torch.device('mps')]:
            with self.subTest(device=device), ExitStack() as stack:
                self.fake_checkpoint(stack)
                cuda = stack.enter_context(patch.object(helpers, 'cuda_allocatable_bytes',
                    side_effect=AssertionError('non-CUDA device')))
                other = stack.enter_context(patch.object(helpers, 'activation_budget_bytes', return_value=7*GIB))
                recommend = stack.enter_context(patch.object(helpers, 'recommended_decode_tiling_config'))
                helpers.tiling_config_for_vae('fake.safetensors', height=1024, width=576,
                    num_frames=185, device=device)
                other.assert_called_once_with(device)
                self.assertEqual(recommend.call_args.kwargs['free_bytes'], 7*GIB)
                cuda.assert_not_called()
        with patch.object(helpers, 'is_diffusion_video_vae', return_value=False), \
             patch.object(helpers, 'cuda_allocatable_bytes', side_effect=AssertionError('Conv path')):
            helpers.tiling_config_for_vae('conv.safetensors', height=1024, width=576,
                num_frames=185, device=torch.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
