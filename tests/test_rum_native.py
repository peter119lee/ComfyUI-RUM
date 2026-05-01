from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rum_native import RUMDiffusersMatchTokenPolicy


def test_diffusers_match_positive_keeps_277_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 277, 8)

    selected = policy.select(cross_attn, "positive")

    assert selected.shape == (1, 277, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()


def test_diffusers_match_positive_recovers_base_and_extra_from_padded_context():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.arange(1 * 589 * 1, dtype=torch.float32).reshape(1, 589, 1)

    selected = policy.select(cross_attn, "positive")

    assert selected.shape == (1, 277, 1)
    assert torch.equal(selected[:, :200], cross_attn[:, :200])
    assert torch.equal(selected[:, 200:], cross_attn[:, -77:])


def test_diffusers_match_negative_keeps_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 512, 8)

    selected = policy.select(cross_attn, "negative")

    assert selected.shape == (1, 512, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()


def test_diffusers_match_negative_crops_overlong_context_to_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.arange(1 * 589 * 1, dtype=torch.float32).reshape(1, 589, 1)

    selected = policy.select(cross_attn, "negative")

    assert selected.shape == (1, 512, 1)
    assert torch.equal(selected, cross_attn[:, :512])


def test_diffusers_match_legacy_does_not_crop_plain_flux2_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 512, 8)

    selected = policy.select(cross_attn, None)

    assert selected.shape == (1, 512, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()
