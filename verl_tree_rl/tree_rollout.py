"""TreeFaithfulRollout: wrapper around verl's vLLM/SGLang rollout.

For Phase 1 (scaffold), only the "flat" pass-through is implemented.
Tree engines (BFS, NegBin, DeepSearch) will be added in Phase 3.

Architecture:
    verl trainer  →  TreeFaithfulRollout.generate_sequences(prompts)
                         │
                         ├── tree_method == "flat":  delegate to inner rollout
                         ├── tree_method == "bfs":   BFS tree engine (Phase 3)
                         ├── tree_method == "negbin": NegBin MCTS engine (Phase 3)
                         └── tree_method == "deepsearch": DeepSearch engine (Phase 3)

The inner rollout (vLLM or SGLang) is created internally with the
same config/model/device_mesh that verl passes to us. We intercept
generate_sequences to run tree search (multiple calls to inner
rollout), then repack the terminal trajectories into a DataProto
that looks identical to what the inner rollout would have returned.
"""

import logging
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__name__)


class TreeFaithfulRollout(BaseRollout):
    """Wraps a standard verl rollout with tree-based generation.

    In "flat" mode, this is a pure pass-through (identical to using
    the inner rollout directly). In tree modes, generate_sequences
    calls the inner rollout multiple times to build a tree, then
    packs terminal trajectories into a single DataProto batch.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # Read tree-specific config from engine_kwargs
        engine_kwargs = getattr(self.config, "engine_kwargs", {}) or {}
        self.tree_method = engine_kwargs.get("tree_method", "flat")
        inner_rollout_name = engine_kwargs.get("inner_rollout_name", "vllm")
        inner_mode = engine_kwargs.get("inner_mode", "sync")

        logger.info(
            f"[TreeFaithfulRollout] tree_method={self.tree_method}, "
            f"inner={inner_rollout_name}/{inner_mode}"
        )

        # Create the inner rollout (owns the actual vLLM/SGLang engine)
        inner_cls = get_rollout_class(inner_rollout_name, inner_mode)
        self.inner_rollout = inner_cls(
            config=config,
            model_config=model_config,
            device_mesh=device_mesh,
        )

        # Tree engine initialization
        self.tree_engine = None
        if self.tree_method == "bfs":
            from verl_tree_rl.tree_engines.bfs_engine import BFSTreeEngine
            self.tree_engine = BFSTreeEngine(engine_kwargs, tokenizer=None)
        elif self.tree_method not in ("flat",):
            raise NotImplementedError(
                f"Tree method '{self.tree_method}' not yet implemented. "
                f"Available: flat, bfs. Coming soon: negbin, deepsearch."
            )

    # ------------------------------------------------------------------
    # Delegate lifecycle methods to inner rollout
    # ------------------------------------------------------------------

    async def resume(self, tags: list[str]):
        return await self.inner_rollout.resume(tags)

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        return await self.inner_rollout.update_weights(weights, **kwargs)

    async def release(self):
        return await self.inner_rollout.release()

    # ------------------------------------------------------------------
    # Core: generate_sequences
    # ------------------------------------------------------------------

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using the selected tree method."""
        # call counter
        if not hasattr(self, "_gen_call_count"):
            self._gen_call_count = 0
        self._gen_call_count += 1
        bsz = prompts.batch.get("input_ids", torch.zeros(0, 0)).shape[0]
        is_validate = prompts.meta_info.get("validate", False)
        do_sample = prompts.meta_info.get("do_sample", True)
        print(f"[TreeRollout] generate_sequences call #{self._gen_call_count} "
              f"bsz={bsz} validate={is_validate} do_sample={do_sample} "
              f"method={self.tree_method}", flush=True)

        if self.tree_method == "flat":
            return self._generate_flat(prompts, **kwargs)
        elif self.tree_method == "bfs":
            return self._generate_bfs(prompts, **kwargs)
        elif self.tree_method == "negbin":
            return self._generate_negbin(prompts, **kwargs)
        elif self.tree_method == "deepsearch":
            return self._generate_deepsearch(prompts, **kwargs)
        else:
            raise ValueError(f"Unknown tree method: {self.tree_method}")

    def _generate_flat(self, prompts: DataProto, **kwargs) -> DataProto:
        """Pure pass-through to inner rollout. Identical output."""
        return self.inner_rollout.generate_sequences(prompts, **kwargs)

    def _generate_bfs(self, prompts: DataProto, **kwargs) -> DataProto:
        """BFS tree rollout: level-wise expansion with fitted branching factors.

        NOTE: verl pre-expands the batch to bsz*n before calling generate_sequences,
        so we return 1 trajectory per input prompt (n_per_prompt=1).
        """
        pad_token_id = prompts.meta_info.get("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0  # fallback
        return self.tree_engine.run(
            prompts=prompts,
            inner_rollout=self.inner_rollout,
            original_response_length=self.config.response_length,
            pad_token_id=int(pad_token_id),
            n_per_prompt=1,
        )

    def _generate_negbin(self, prompts: DataProto, **kwargs) -> DataProto:
        raise NotImplementedError("NegBin MCTS engine not yet implemented (Phase 3)")

    def _generate_deepsearch(self, prompts: DataProto, **kwargs) -> DataProto:
        raise NotImplementedError("DeepSearch engine not yet implemented (Phase 3)")
