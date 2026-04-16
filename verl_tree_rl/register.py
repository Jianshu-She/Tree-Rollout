"""Register TreeFaithfulRollout into verl's rollout registry.

Usage: import this module before verl starts. The simplest way is to
add `import verl_tree_rl.register` to the training script or set
PYTHONPATH to include the parent of verl_tree_rl/.

This adds ("tree_faithful", "sync") to verl's _ROLLOUT_REGISTRY so
that setting `actor_rollout_ref.rollout.name=tree_faithful` in the
hydra config will route rollout creation to our TreeFaithfulRollout.
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY

_ROLLOUT_REGISTRY[("tree_faithful", "sync")] = (
    "verl_tree_rl.tree_rollout.TreeFaithfulRollout"
)

print("[verl_tree_rl] Registered tree_faithful rollout into verl registry")
