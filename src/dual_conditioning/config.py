"""Validated YAML configuration for generation and evaluation.

The frozen experiment used environment variables for several guidance choices.
This module makes those choices explicit and rejects unknown values instead of
silently falling back to a different guidance field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GuidanceConfig:
    space: str
    field: str
    alpha: float
    radius_angstrom: float
    clip_angstrom: float
    stop_step: int | None = None

    def validate(self) -> None:
        if self.space not in {"x0", "z"}:
            raise ValueError(f"guide space must be 'x0' or 'z', got {self.space!r}")
        if self.field not in {"shape", "protrusion"}:
            raise ValueError(
                f"guide field must be 'shape' or 'protrusion', got {self.field!r}"
            )
        if self.alpha <= 0:
            raise ValueError("guide alpha must be positive")
        if self.radius_angstrom <= 0:
            raise ValueError("guide radius must be positive")
        if self.clip_angstrom <= 0:
            raise ValueError("guide clip must be positive")
        if self.stop_step is not None and self.stop_step < 0:
            raise ValueError("stop_step must be non-negative")


@dataclass(frozen=True)
class SamplingConfig:
    timesteps: int
    resamplings: int
    samples_per_run: int
    center: str

    def validate(self) -> None:
        if self.timesteps <= 0 or self.resamplings <= 0 or self.samples_per_run <= 0:
            raise ValueError("timesteps, resamplings, and samples_per_run must be positive")
        if self.center not in {"ligand", "pocket"}:
            raise ValueError("center must be 'ligand' or 'pocket'")


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    lambdas: tuple[float, ...]
    sampling: SamplingConfig
    guidance: GuidanceConfig

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.lambdas:
            raise ValueError("at least one guidance strength is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("guidance strengths must be unique")
        if any(value < 0 for value in self.lambdas):
            raise ValueError("guidance strengths must be non-negative")
        self.sampling.validate()
        self.guidance.validate()


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing configuration key: {key}")
    return mapping[key]


def load_campaign_config(path: str | Path) -> CampaignConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    sampling_raw = _require(raw, "sampling")
    guidance_raw = _require(raw, "guidance")
    if not isinstance(sampling_raw, dict) or not isinstance(guidance_raw, dict):
        raise ValueError("sampling and guidance must be mappings")

    config = CampaignConfig(
        seeds=tuple(int(value) for value in _require(raw, "seeds")),
        lambdas=tuple(float(value) for value in _require(raw, "lambdas")),
        sampling=SamplingConfig(
            timesteps=int(_require(sampling_raw, "timesteps")),
            resamplings=int(_require(sampling_raw, "resamplings")),
            samples_per_run=int(_require(sampling_raw, "samples_per_run")),
            center=str(_require(sampling_raw, "center")),
        ),
        guidance=GuidanceConfig(
            space=str(_require(guidance_raw, "space")),
            field=str(_require(guidance_raw, "field")),
            alpha=float(_require(guidance_raw, "alpha")),
            radius_angstrom=float(_require(guidance_raw, "radius_angstrom")),
            clip_angstrom=float(_require(guidance_raw, "clip_angstrom")),
            stop_step=(
                None
                if guidance_raw.get("stop_step") is None
                else int(guidance_raw["stop_step"])
            ),
        ),
    )
    config.validate()
    return config
