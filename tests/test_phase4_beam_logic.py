#!/usr/bin/env python3
"""Pure-logic regression tests for variable-increment greedy/beam search."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
GROWTH = PROJECT / "src" / "growth"
sys.path.insert(0, str(GROWTH))

import orchestrator_beam as beam


def test_tradeoff_gate() -> None:
    ok, gain, g_t, g_p = beam.compute_gate(0.49, 0.40, 0.50, 0.41, 0.01)
    assert ok
    assert np.isclose(gain, 0.02)
    assert np.isclose(g_t, 0.01)
    assert np.isclose(g_p, 0.01)

    ok, *_ = beam.compute_gate(0.505, 0.39, 0.50, 0.41, 0.01)
    assert ok, "A small loss in one metric may pass when net gain is positive"

    ok, *_ = beam.compute_gate(0.52, 0.37, 0.50, 0.41, 0.01)
    assert not ok, "Per-objective deterioration beyond epsilon must fail"


def test_absolute_quality_ranking() -> None:
    candidates = [
        {
            "name": "better_absolute",
            "absolute_quality": beam.absolute_quality(0.40, 0.30),
            "grown_xyz": np.array([[0.0, 0.0, 0.0]]),
        },
        {
            "name": "worse_absolute",
            "absolute_quality": beam.absolute_quality(0.45, 0.35),
            "grown_xyz": np.array([[2.0, 0.0, 0.0]]),
        },
    ]
    chosen = beam.select_beams(candidates, k=1, chamfer_min=0.75)
    assert chosen[0]["name"] == "better_absolute"


def test_chamfer_diversity() -> None:
    first = {
        "name": "first",
        "absolute_quality": -0.5,
        "grown_xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    near = {
        "name": "near",
        "absolute_quality": -0.6,
        "grown_xyz": np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]),
    }
    far = {
        "name": "far",
        "absolute_quality": -0.7,
        "grown_xyz": np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    }
    chosen = beam.select_beams([first, near, far], k=3, chamfer_min=0.75)
    assert [item["name"] for item in chosen] == ["first", "far"]


def test_bounded_graduation() -> None:
    assert beam.graduation_status(14, target=16, tol=1) == "growing"
    assert beam.graduation_status(15, target=16, tol=1) == "graduated"
    assert beam.graduation_status(17, target=16, tol=1) == "graduated"
    assert beam.graduation_status(18, target=16, tol=1) == "oversize"


def test_parent_retention() -> None:
    parent = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    retained = np.array(
        [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    lost = np.array([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert beam.scaffold_retained(parent, retained, tol=0.5)
    assert not beam.scaffold_retained(parent, lost, tol=0.5)


def test_checkpoint_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        scaffold = out / "anchor.sdf"
        scaffold.write_text("placeholder\n", encoding="utf-8")
        beams = [
            {
                "fix_atoms_arg": str(scaffold),
                "scaffold_is_sdf": True,
                "tani_B": 0.5,
                "prot_B": 0.4,
                "anchor_heavy": 12,
                "lineage": "root>s1b0",
                "cumulative_gain": 0.2,
            }
        ]
        beam.save_checkpoint(
            out,
            stage_completed=2,
            beams=beams,
            graduated_records=[],
            traj_rows=[{"stage": 2}],
            dead_count=0,
            config={"k": 1},
        )
        loaded = beam.load_checkpoint(out)
        assert loaded is not None
        start_stage, restored, graduated, rows, dead = loaded
        assert start_stage == 3
        assert restored[0]["fix_atoms_arg"] == str(scaffold)
        assert restored[0]["lineage"] == "root>s1b0"
        assert graduated == []
        assert rows == [{"stage": 2}]
        assert dead == 0
        json.loads((out / "beam_state.json").read_text())


def main() -> None:
    tests = [
        test_tradeoff_gate,
        test_absolute_quality_ranking,
        test_chamfer_diversity,
        test_bounded_graduation,
        test_parent_retention,
        test_checkpoint_roundtrip,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\nAll {len(tests)} Phase-4 beam tests passed.")


if __name__ == "__main__":
    main()
