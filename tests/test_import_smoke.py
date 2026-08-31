"""Import-and-construct smoke test.

The ea948e0 merge silently dropped seven `INTENT_*` constants from `src/config.py`
while `src/dialogue.py` continued to import them, breaking every entry point
(evaluator, chat, scripts) with an ImportError at import time. No test in the
suite exercised the top-level import path, so the regression escaped CI.

This module owns exactly one job: import `starter.agent.Agent` (the submission
entry point) and construct it against a tiny in-memory catalog. If any module
in the chain fails to import (missing constants, broken relative import, syntax
error) this test fails immediately and unambiguously.

Keep the catalog tiny and set `Agent.USE_VECTOR = False` to avoid loading
`cache/embeddings.npy`, which is a large binary artifact not present in fresh
checkouts. The goal is not to exercise ranking — it is to catch import breakage.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ImportSmokeTest(unittest.TestCase):
    def test_import_agent_module(self):
        # Direct import from the submission entry point the evaluator uses.
        from starter.agent import Agent  # noqa: F401
        self.assertTrue(callable(Agent))

    def test_import_src_agent(self):
        # Direct import of the implementation module, exercising the src.config
        # import chain that broke in ea948e0.
        from src.agent import Agent  # noqa: F401
        self.assertTrue(callable(Agent))

    def test_construct_agent_with_minimal_catalog(self):
        from starter.agent import Agent
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": "SMOKE_A",
                    "title": "waterproof leather hiking boot",
                    "features": ["waterproof"],
                    "details": {"size": "10"},
                    "description": ["rugged boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "BootCo",
                    "average_rating": 4.5,
                    "rating_number": 500,
                    "price": 89.0,
                },
                {
                    "parent_asin": "SMOKE_B",
                    "title": "canvas sneaker",
                    "features": [],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "store": "SneakerCo",
                    "average_rating": 4.0,
                    "rating_number": 100,
                    "price": 49.0,
                },
            ]
            catalog_path.write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
            )
            # USE_VECTOR=False avoids requiring cache/embeddings.npy.
            old_vector = Agent.USE_VECTOR
            Agent.USE_VECTOR = False
            try:
                agent = Agent(str(catalog_path), dcp_state_dir=tmpdir, persist_dcp=False)
                self.assertIsNotNone(agent)
            finally:
                Agent.USE_VECTOR = old_vector


if __name__ == "__main__":
    unittest.main()
