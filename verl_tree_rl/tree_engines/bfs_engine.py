"""BFS Tree engine for verl GRPO training.

Generates trajectories by level-wise tree expansion with fitted branching
factors. At each depth, non-terminal frontier nodes are expanded in batch
by calling the inner rollout's generate_sequences with a short
max_new_tokens (tokens_per_step). Terminal trajectories are assembled by
concatenating chunks along root-to-leaf paths.

Uses DeepSearch-style DataProto manipulation (pad/stack/slice) for
compatibility with verl's tensor pipeline.
"""

import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataProto helpers (adapted from DeepSearch utils/data_proto.py)
# ---------------------------------------------------------------------------

def _slice_dp(dp: DataProto, start: int, end: int) -> DataProto:
    d = {}
    for k, v in dp.batch.items():
        d[k] = v[start:end]
    for k, v in dp.non_tensor_batch.items():
        d[k] = np.array(v[start:end], dtype=object)
    return DataProto.from_single_dict(d, meta_info=dp.meta_info)


def _stack_dps(dps: List[DataProto]) -> DataProto:
    tensor_data = defaultdict(list)
    non_tensor_data = defaultdict(list)
    for dp in dps:
        for k, v in dp.batch.items():
            tensor_data[k].append(v if v.dim() >= 2 else v.unsqueeze(0))
        for k, v in dp.non_tensor_batch.items():
            non_tensor_data[k].extend(v)
    batch = {k: torch.cat(vs, dim=0) for k, vs in tensor_data.items()}
    ntb = {k: np.array(vs, dtype=object) for k, vs in non_tensor_data.items()}
    meta = dps[0].meta_info if dps else None
    return DataProto.from_single_dict({**batch, **ntb}, meta_info=meta)


def _pad_dps(dps: List[DataProto], pad_token_id: int) -> List[DataProto]:
    """Left-pad all DataProtos to the same sequence length per key."""
    dps = [deepcopy(dp) for dp in dps]
    for key in dps[0].batch.keys():
        max_len = max(dp.batch[key].shape[-1] for dp in dps)
        pad_id = 0 if key in ("attention_mask", "rollout_log_probs") else pad_token_id
        left = key in ("input_ids", "prompts", "attention_mask", "position_ids")
        for dp in dps:
            if dp.batch[key].shape[-1] < max_len:
                dp.batch[key] = pad_sequence_to_length(
                    dp.batch[key], max_len, pad_id, left_pad=left,
                )
                if key == "position_ids" and not left:
                    cur = dp.batch[key]
                    base = cur[0, -1].item()
                    delta = max_len - cur.shape[1]
                    ext = torch.arange(base + 1, base + delta + 1,
                                       device=cur.device).expand(cur.shape[0], -1)
                    dp.batch[key] = torch.cat([cur, ext], dim=-1)
    return dps


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("uid", "depth", "delta", "children", "is_terminal", "parent")

    def __init__(self, uid: str, depth: int, delta: Optional[DataProto],
                 parent: Optional["_Node"] = None):
        self.uid = uid
        self.depth = depth
        self.delta = delta          # DataProto chunk for this node's generation
        self.children: List[_Node] = []
        self.is_terminal = False
        self.parent = parent

    def full_trajectory_deltas(self) -> List[DataProto]:
        """Collect deltas from root to this node (inclusive)."""
        path = []
        node = self
        while node is not None and node.delta is not None:
            path.append(node.delta)
            node = node.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# BFS Engine
# ---------------------------------------------------------------------------

class BFSTreeEngine:
    """Level-wise BFS tree expansion for verl rollout.

    Config keys (passed via engine_kwargs):
        fitted_params_path: str  — path to fitted_parameters.json
        training_stage: str      — e.g. "step_0"
        tokens_per_step: int     — max new tokens per expansion (default 256)
        max_depth: int           — max tree depth (default 12)
    """

    def __init__(self, config: Dict[str, Any], tokenizer):
        self.tokens_per_step = int(config.get("tokens_per_step", 256))
        self.max_depth = int(config.get("max_depth", 12))
        self.tokenizer = tokenizer

        # Load branching factors
        params_path = config.get("fitted_params_path", "")
        stage = config.get("training_stage", "step_0")
        if params_path:
            with open(params_path) as f:
                all_params = json.load(f)
            stage_params = all_params.get(stage, {})
            self.branching_factors = {}
            for d_str, entry in stage_params.items():
                lam = entry.get("poisson_lambda", 1.0)
                self.branching_factors[int(d_str)] = max(1, round(lam))
        else:
            self.branching_factors = {0: 8, 1: 2, 2: 1}
            logger.warning("No fitted_params_path; using default branching [8,2,1]")

        logger.info(f"[BFSTreeEngine] branching={self.branching_factors}, "
                    f"tokens_per_step={self.tokens_per_step}, max_depth={self.max_depth}")

    def get_bf(self, depth: int) -> int:
        return self.branching_factors.get(depth, 1)

    def run(
        self,
        prompts: DataProto,
        inner_rollout,
        original_response_length: int,
        pad_token_id: int,
        n_per_prompt: int,
    ) -> DataProto:
        """Run BFS tree expansion on a batch of prompts.

        Args:
            prompts: input batch [bsz, prompt_len]
            inner_rollout: verl rollout handle with generate_sequences()
            original_response_length: the full response length from config
            pad_token_id: tokenizer pad token id
            n_per_prompt: target number of trajectories per prompt

        Returns:
            DataProto with [bsz * n_per_prompt, prompt_len + response_len]
        """
        bsz = prompts.batch["input_ids"].shape[0]
        prompt_len = prompts.batch["input_ids"].shape[1]

        # For each prompt, build a tree
        all_terminal_dps = []

        for i in range(bsz):
            single_prompt = _slice_dp(prompts, i, i + 1)
            terminals = self._build_tree(single_prompt, inner_rollout,
                                         pad_token_id)

            # Sample/pad to n_per_prompt
            if len(terminals) > n_per_prompt:
                terminals = terminals[:n_per_prompt]
            elif len(terminals) < n_per_prompt:
                # Duplicate last terminal to pad
                while len(terminals) < n_per_prompt:
                    terminals.append(deepcopy(terminals[-1]))

            all_terminal_dps.extend(terminals)

        # Pad all to same length, then stack
        all_terminal_dps = _pad_dps(all_terminal_dps, pad_token_id)
        result = _stack_dps(all_terminal_dps)
        return result

    def _build_tree(
        self,
        single_prompt: DataProto,
        inner_rollout,
        pad_token_id: int,
    ) -> List[DataProto]:
        """Build BFS tree for one prompt. Returns list of terminal DataProtos."""
        uid = single_prompt.non_tensor_batch.get("uid", np.array(["0"]))[0]
        root = _Node(uid=str(uid), depth=0, delta=None)
        frontier = [root]
        terminals: List[_Node] = []

        for depth in range(self.max_depth):
            if not frontier:
                break
            bf = self.get_bf(depth)
            if bf <= 0:
                break

            # Build prompts for all frontier nodes
            batch_prompts = []
            batch_nodes = []
            for node in frontier:
                # Construct full prompt = original + all ancestor deltas
                prompt_dp = self._assemble_prompt(single_prompt, node)
                for _ in range(bf):
                    batch_prompts.append(deepcopy(prompt_dp))
                    batch_nodes.append(node)

            if not batch_prompts:
                break

            # Pad and stack
            batch_prompts = _pad_dps(batch_prompts, pad_token_id)
            gen_input = _stack_dps(batch_prompts)

            # Generate short chunks
            saved_resp_len = inner_rollout.config.response_length
            inner_rollout.config.response_length = self.tokens_per_step
            try:
                outputs = inner_rollout.generate_sequences(gen_input)
            finally:
                inner_rollout.config.response_length = saved_resp_len

            # Process outputs: create child nodes
            next_frontier = []
            for j, parent_node in enumerate(batch_nodes):
                child_dp = _slice_dp(outputs, j, j + 1)

                # Check if terminal (EOS in response or max depth)
                response_ids = child_dp.batch.get("responses")
                has_eos = False
                if response_ids is not None:
                    attn = child_dp.batch.get("attention_mask")
                    if attn is not None and attn.shape[-1] > 0:
                        # response part of attention mask: 0 means padding (after EOS)
                        resp_attn = attn[0, -response_ids.shape[1]:]
                        has_eos = resp_attn[-1].item() == 0

                child = _Node(
                    uid=f"{parent_node.uid}_d{depth}_c{j}",
                    depth=depth + 1,
                    delta=child_dp,
                    parent=parent_node,
                )
                child.is_terminal = has_eos or (depth + 1 >= self.max_depth)
                parent_node.children.append(child)

                if child.is_terminal:
                    terminals.append(child)
                else:
                    next_frontier.append(child)

            frontier = next_frontier

        # Convert terminal nodes to full DataProtos
        result = []
        for node in terminals:
            full_dp = self._assemble_full_trajectory(single_prompt, node)
            result.append(full_dp)

        if not result:
            # Fallback: return the prompt with empty response
            logger.warning("BFS produced no terminals; returning empty response")
            result = [single_prompt]

        return result

    def _assemble_prompt(self, original_prompt: DataProto, node: _Node) -> DataProto:
        """Construct input for generation = original prompt + accumulated response chunks."""
        if node.depth == 0:
            return deepcopy(original_prompt)

        # Gather response chunks from root to this node
        deltas = node.full_trajectory_deltas()
        if not deltas:
            return deepcopy(original_prompt)

        # Concatenate original prompt input_ids with delta response tokens
        prompt_ids = original_prompt.batch["input_ids"]  # [1, prompt_len]

        response_chunks = []
        for d in deltas:
            resp = d.batch.get("responses")
            if resp is not None:
                # Remove padding (keep only non-pad tokens)
                attn = d.batch.get("attention_mask")
                if attn is not None:
                    resp_attn = attn[0, -resp.shape[1]:]
                    valid = resp_attn.bool()
                    response_chunks.append(resp[0][valid].unsqueeze(0))
                else:
                    response_chunks.append(resp)

        if response_chunks:
            all_resp = torch.cat(response_chunks, dim=-1)
            new_ids = torch.cat([prompt_ids, all_resp], dim=-1)
        else:
            new_ids = prompt_ids

        seq_len = new_ids.shape[1]
        new_attn = torch.ones_like(new_ids)
        new_pos = torch.arange(seq_len, device=new_ids.device).unsqueeze(0)

        d = {
            "input_ids": new_ids,
            "attention_mask": new_attn,
            "position_ids": new_pos,
        }
        # Copy non_tensor_batch
        for k, v in original_prompt.non_tensor_batch.items():
            d[k] = v
        return DataProto.from_single_dict(d, meta_info=original_prompt.meta_info)

    def _assemble_full_trajectory(
        self, original_prompt: DataProto, terminal_node: _Node
    ) -> DataProto:
        """Assemble the full trajectory DataProto for a terminal node."""
        # This is the same as _assemble_prompt but returns the full
        # prompt + response in the format verl expects
        prompt_ids = original_prompt.batch["input_ids"]  # [1, prompt_len]
        prompt_len = prompt_ids.shape[1]

        # Gather all response chunks
        deltas = terminal_node.full_trajectory_deltas()
        response_chunks = []
        for d in deltas:
            resp = d.batch.get("responses")
            if resp is not None:
                attn = d.batch.get("attention_mask")
                if attn is not None:
                    resp_attn = attn[0, -resp.shape[1]:]
                    valid = resp_attn.bool()
                    response_chunks.append(resp[0][valid].unsqueeze(0))
                else:
                    response_chunks.append(resp)

        if response_chunks:
            full_response = torch.cat(response_chunks, dim=-1)
        else:
            full_response = torch.zeros(1, 1, dtype=prompt_ids.dtype,
                                        device=prompt_ids.device)

        full_ids = torch.cat([prompt_ids, full_response], dim=-1)
        seq_len = full_ids.shape[1]
        resp_len = full_response.shape[1]

        attn = torch.ones_like(full_ids)
        pos = torch.arange(seq_len, device=full_ids.device).unsqueeze(0)

        d = {
            "prompts": prompt_ids,
            "responses": full_response,
            "input_ids": full_ids,
            "attention_mask": attn,
            "position_ids": pos,
        }
        for k, v in original_prompt.non_tensor_batch.items():
            d[k] = v
        return DataProto.from_single_dict(d, meta_info=original_prompt.meta_info)
