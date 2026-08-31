"""Shared benchmark setup.

Official scoring treats sessions as isolated.  This helper gives every evaluation invocation a
fresh ephemeral DCP directory, disables DCP persistence, and pins the candidate pool independently
of popularity/profile ablations.  The temporary-directory object is held by the returned Agent so
it remains alive for the complete evaluation and is cleaned up when the process exits.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.agent import Agent


def new_isolated_agent(
    catalog_path: str | Path, pool_size: int = 200, agent_cls: type[Agent] = Agent,
) -> Agent:
    """Create an evaluation Agent with no cross-session durable state."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    state_dir = tempfile.TemporaryDirectory(prefix="techjam-eval-dcp-")
    try:
        agent = agent_cls(catalog_path, dcp_state_dir=state_dir.name, persist_dcp=False)
    except TypeError:
        # Preserve compatibility with third-party agents that implement only the official
        # one-argument constructor.  Their own state cannot be redirected, but the flags below
        # still disable DCP when the implementation exposes them.
        agent = agent_cls(catalog_path)
    # Instance flags avoid changing normal Agent defaults or leaking benchmark policy to callers.
    agent.USE_DCP = False
    agent.DCP_DISTILL = False
    agent.DCP_PROFILE = False
    agent.DCP_ORCHESTRATION = False
    agent.DCP_GUIDANCE_LEARNING = False
    agent.DCP_PERSISTENCE = False
    agent.POOL_SIZE_OVERRIDE = pool_size
    agent._benchmark_state_dir = state_dir  # keep TemporaryDirectory alive for this run
    return agent
