"""Regression guard for the SATISFACTION_* runtime-wiring bug (see EXPERIMENT_LOG.md P1.2).

Before the fix, `_satisfaction` was built once in Agent.__init__ and captured its knobs into its
own instance state. Mutating `agent.SATISFACTION_SEM_GATE_HIGH` on the Agent afterwards did NOT
reach the scorer's `sem_gate_high`, so sweep harnesses and eval_matrix's `apply_config` were
silently ignoring every SATISFACTION_* override. These asserts pin the invariant.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make `src` importable when the test file is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent  # noqa: E402


class SatisfactionWiringTest(unittest.TestCase):
    """Two-row catalog is enough — we only exercise the wiring, not the ranking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._tmpdir.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "waterproof hiking boot", "features": [],
             "details": {}, "description": [], "categories": [], "store": "",
             "average_rating": 4.5, "rating_number": 500, "price": 89.0},
            {"parent_asin": "B", "title": "canvas sneaker", "features": [],
             "details": {}, "description": [], "categories": [], "store": "",
             "average_rating": 4.0, "rating_number": 100, "price": 49.0},
        ]
        catalog_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        Agent.USE_VECTOR = False  # skip real embeddings cache
        cls.agent = Agent(str(catalog_path))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_public_scorer_alias_exists(self) -> None:
        # `satisfaction_scorer` is the public alias `_satisfaction` — the attribute name that
        # sweep scripts and this test rely on.
        self.assertIs(self.agent.satisfaction_scorer, self.agent._satisfaction)

    def test_runtime_override_propagates_after_refresh(self) -> None:
        # The core invariant the wiring fix restores. Overriding on the Agent + refreshing must
        # reach the scorer's captured attrs. Without refresh_satisfaction_scorer() the assert
        # below (sem_gate_high == 0.65) would have failed against the stale 0.85.
        self.agent.SATISFACTION_SEM_GATE_HIGH = 0.65
        self.agent.SATISFACTION_SEM_GATE_LOW = 0.25
        self.agent.SATISFACTION_POP_WEIGHT = 0.10
        self.agent.refresh_satisfaction_scorer()

        self.assertAlmostEqual(self.agent.satisfaction_scorer.sem_gate_high, 0.65)
        self.assertAlmostEqual(self.agent.satisfaction_scorer.sem_gate_low, 0.25)
        self.assertAlmostEqual(self.agent.satisfaction_scorer.pop_weight, 0.10)

    def test_gate_math_reflects_new_thresholds(self) -> None:
        # Not just the attribute but the gate function itself uses the new threshold.
        self.agent.SATISFACTION_SEM_GATE_LOW = 0.25
        self.agent.SATISFACTION_SEM_GATE_HIGH = 0.65
        self.agent.refresh_satisfaction_scorer()
        gate = self.agent.satisfaction_scorer._sem_gate
        self.assertEqual(gate(0.20), 1.0)   # below LOW → full popularity
        self.assertEqual(gate(0.65), 0.0)   # at HIGH → zero popularity
        self.assertAlmostEqual(gate(0.45), 0.5, places=3)  # midpoint of the ramp


if __name__ == "__main__":
    unittest.main()
