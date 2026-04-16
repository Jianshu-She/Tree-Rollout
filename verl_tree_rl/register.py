"""Register TreeFaithfulRollout into verl's rollout registry.

NOTE: As of 2026-04-16, we add the registry entry directly to
verl/workers/rollout/base.py instead of monkey-patching at runtime.
This ensures Ray worker processes (which are separate from the main
process) also see the registration.

This file is kept for documentation and for the pre-flight check
in run_grpo_flat.sh.
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY

assert ("tree_faithful", "sync") in _ROLLOUT_REGISTRY, (
    "tree_faithful not found in verl's rollout registry. "
    "Please add the entry to verl/workers/rollout/base.py"
)

print("[verl_tree_rl] Confirmed: tree_faithful is registered in verl")
