from __future__ import annotations

from pathlib import Path

import pytest

from dual_conditioning.config import load_campaign_config


def test_exp06_configuration_is_explicit_and_valid():
    config = load_campaign_config(Path("configs/exp06_generation.yaml"))
    assert config.seeds == (1101, 2202, 3303, 4404, 5505)
    assert config.lambdas == (0.0, 20.0, 50.0, 100.0, 200.0)
    assert config.guidance.space == "x0"
    assert config.guidance.field == "shape"
    assert config.guidance.alpha == 0.3


def test_unknown_guidance_field_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("""
seeds: [1]
lambdas: [0]
sampling: {timesteps: 1, resamplings: 1, samples_per_run: 1, center: ligand}
guidance: {space: x0, field: typo, alpha: 0.3, radius_angstrom: 1.7, clip_angstrom: 1.0}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="guide field"):
        load_campaign_config(path)
