"""Abstract base class for tree engines.

Each tree engine takes a batch of prompts and an inner rollout handle,
runs its tree search algorithm (making multiple generate_sequences calls
to the inner rollout), and returns a DataProto that looks like a normal
rollout output (prompts + responses + masks).

Phase 3 will implement: BFSEngine, NegBinEngine, DeepSearchEngine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from verl import DataProto


class BaseTreeEngine(ABC):
    """Abstract tree engine for verl_tree_rl.

    Subclasses implement `run()` which takes prompts + inner_rollout
    and returns a repacked DataProto with terminal trajectories.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with tree-specific config from engine_kwargs."""
        self.config = config

    @abstractmethod
    def run(
        self,
        prompts: DataProto,
        inner_rollout,
        **kwargs,
    ) -> DataProto:
        """Run tree search on a batch of prompts.

        Args:
            prompts: Input DataProto batch from verl trainer.
            inner_rollout: The underlying vLLM/SGLang rollout handle.
                          Call inner_rollout.generate_sequences(partial_prompts)
                          to expand tree nodes.
            **kwargs: Additional arguments (e.g., max_new_tokens overrides).

        Returns:
            DataProto with the same schema as inner_rollout.generate_sequences
            would return, but containing terminal trajectories from the tree
            instead of independent flat samples.

        The returned DataProto must contain at minimum:
            batch["prompts"]:        [bsz * n, prompt_length]
            batch["responses"]:      [bsz * n, response_length]
            batch["input_ids"]:      [bsz * n, prompt_length + response_length]
            batch["attention_mask"]: [bsz * n, prompt_length + response_length]
            batch["position_ids"]:   [bsz * n, prompt_length + response_length]

        where n = number of terminal trajectories per prompt (may vary
        per prompt for tree methods; padding is needed to make a batch).
        """
        raise NotImplementedError
