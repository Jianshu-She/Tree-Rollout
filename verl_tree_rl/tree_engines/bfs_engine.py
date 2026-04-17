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

        Returns DataProto with FIXED shapes matching flat rollout output:
            prompts:        [bsz * n_per_prompt, prompt_len]
            responses:      [bsz * n_per_prompt, response_len]  (right-padded)
            input_ids:      [bsz * n_per_prompt, prompt_len + response_len]
            attention_mask:  [bsz * n_per_prompt, prompt_len + response_len]
            position_ids:   [bsz * n_per_prompt, prompt_len + response_len]
        """
        if pad_token_id is None:
            pad_token_id = 0
        pad_token_id = int(pad_token_id)
        bsz = prompts.batch["input_ids"].shape[0]
        prompt_len = prompts.batch["input_ids"].shape[1]
        resp_len = original_response_length
        total_len = prompt_len + resp_len
        device = prompts.batch["input_ids"].device

        all_prompts = []
        all_responses = []
        all_input_ids = []
        all_attn = []
        all_pos = []
        all_ntb = {k: [] for k in prompts.non_tensor_batch.keys()}

        for i in range(bsz):
            single_prompt = _slice_dp(prompts, i, i + 1)
            terminal_nodes = self._build_tree_nodes(single_prompt, inner_rollout,
                                                     pad_token_id)

            # Sample/pad to n_per_prompt
            if len(terminal_nodes) > n_per_prompt:
                terminal_nodes = terminal_nodes[:n_per_prompt]
            while len(terminal_nodes) < n_per_prompt:
                terminal_nodes.append(terminal_nodes[-1] if terminal_nodes
                                      else _Node("empty", 0, None))

            prompt_ids = single_prompt.batch["input_ids"]  # [1, prompt_len]

            for node in terminal_nodes:
                # Get full response token ids from root-to-leaf
                resp_tokens = self._get_response_tokens(node, device)

                # Truncate or pad response to resp_len
                if resp_tokens.shape[1] > resp_len:
                    resp_tokens = resp_tokens[:, :resp_len]
                elif resp_tokens.shape[1] < resp_len:
                    pad_size = resp_len - resp_tokens.shape[1]
                    padding = torch.full((1, pad_size), pad_token_id,
                                         dtype=resp_tokens.dtype, device=device)
                    resp_tokens = torch.cat([resp_tokens, padding], dim=-1)

                full_ids = torch.cat([prompt_ids, resp_tokens], dim=-1)
                # attention_mask: 1 for real tokens, 0 for padding
                resp_real_len = min(self._get_response_tokens(node, device).shape[1],
                                    resp_len)
                attn_resp = torch.cat([
                    torch.ones(1, resp_real_len, dtype=torch.long, device=device),
                    torch.zeros(1, resp_len - resp_real_len, dtype=torch.long, device=device),
                ], dim=-1)
                attn = torch.cat([single_prompt.batch["attention_mask"], attn_resp], dim=-1)
                # position_ids: preserve prompt's position_ids (for left-pad handling),
                # continue from last prompt position for response
                prompt_pos = single_prompt.batch["position_ids"]  # [1, prompt_len]
                base_pos = prompt_pos[0, -1].item()
                resp_pos = torch.arange(
                    base_pos + 1, base_pos + 1 + resp_len, device=device
                ).unsqueeze(0)
                pos = torch.cat([prompt_pos, resp_pos], dim=-1)

                all_prompts.append(prompt_ids)
                all_responses.append(resp_tokens)
                all_input_ids.append(full_ids)
                all_attn.append(attn)
                all_pos.append(pos)
                for k in all_ntb:
                    all_ntb[k].append(single_prompt.non_tensor_batch[k][0])

        result_dict = {
            "prompts": torch.cat(all_prompts, dim=0),
            "responses": torch.cat(all_responses, dim=0),
            "input_ids": torch.cat(all_input_ids, dim=0),
            "attention_mask": torch.cat(all_attn, dim=0),
            "position_ids": torch.cat(all_pos, dim=0),
        }
        for k, v in all_ntb.items():
            result_dict[k] = np.array(v, dtype=object)

        return DataProto.from_single_dict(result_dict, meta_info=prompts.meta_info)

    def _get_response_tokens(self, node: _Node, device) -> torch.Tensor:
        """Extract concatenated response tokens from root-to-leaf path."""
        if node.delta is None:
            return torch.zeros(1, 0, dtype=torch.long, device=device)
        deltas = node.full_trajectory_deltas()
        chunks = []
        for d in deltas:
            resp = d.batch.get("responses")
            if resp is None:
                continue
            attn = d.batch.get("attention_mask")
            if attn is not None and resp.shape[1] <= attn.shape[1]:
                resp_attn = attn[0, -resp.shape[1]:]
                valid = resp_attn.bool()
                chunks.append(resp[0][valid].unsqueeze(0).to(device))
            else:
                chunks.append(resp.to(device))
        if not chunks:
            return torch.zeros(1, 0, dtype=torch.long, device=device)
        return torch.cat(chunks, dim=-1)

    def _build_tree_nodes(
        self,
        single_prompt: DataProto,
        inner_rollout,
        pad_token_id: int,
    ) -> List[_Node]:
        """Build BFS tree for one prompt. Returns list of terminal _Node objects."""
        uid = single_prompt.non_tensor_batch.get("uid", np.array(["0"]))[0]
        root = _Node(uid=str(uid), depth=0, delta=None)
        frontier = [root]
        terminals: List[_Node] = []

        print(f"[BFS] start tree for uid={uid} max_depth={self.max_depth}", flush=True)

        for depth in range(self.max_depth):
            if not frontier:
                print(f"[BFS] depth={depth}: empty frontier, stopping", flush=True)
                break
            bf = self.get_bf(depth)
            if bf <= 0:
                print(f"[BFS] depth={depth}: bf={bf}, stopping", flush=True)
                break
            print(f"[BFS] depth={depth}: frontier={len(frontier)} bf={bf} terminals_so_far={len(terminals)}", flush=True)

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

            print(f"[BFS] depth={depth}: padding {len(batch_prompts)} prompts", flush=True)
            # Pad and stack
            batch_prompts = _pad_dps(batch_prompts, pad_token_id)
            gen_input = _stack_dps(batch_prompts)
            print(f"[BFS] depth={depth}: calling generate_sequences with bsz={gen_input.batch['input_ids'].shape[0]}", flush=True)

            # Generate short chunks — override sampling_params.max_tokens directly
            # (sampling_params is cached at init in vLLMRollout.__init__)
            saved_max_tokens = getattr(inner_rollout.sampling_params, "max_tokens", None)
            try:
                inner_rollout.sampling_params.max_tokens = self.tokens_per_step
                outputs = inner_rollout.generate_sequences(gen_input)
            finally:
                if saved_max_tokens is not None:
                    inner_rollout.sampling_params.max_tokens = saved_max_tokens
            print(f"[BFS] depth={depth}: got outputs, response shape={outputs.batch['responses'].shape}", flush=True)

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
            print(f"[BFS] depth={depth} done: new_frontier={len(frontier)} terminals={len(terminals)}", flush=True)

        print(f"[BFS] tree done: total terminals={len(terminals)} frontier={len(frontier)}", flush=True)
        # If we stopped with frontier non-empty (hit max_depth), force them terminal
        for node in frontier:
            if not node.is_terminal:
                node.is_terminal = True
                terminals.append(node)

        if not terminals:
            logger.warning("BFS produced no terminals; returning root node")
            terminals = [root]

        print(f"[BFS] returning {len(terminals)} terminal nodes", flush=True)
        return terminals

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

        device = prompt_ids.device
        if response_chunks:
            all_resp = torch.cat([c.to(device) for c in response_chunks], dim=-1)
            new_ids = torch.cat([prompt_ids, all_resp], dim=-1)
        else:
            new_ids = prompt_ids

        seq_len = new_ids.shape[1]
        new_attn = torch.ones_like(new_ids)
        new_pos = torch.arange(seq_len, device=device).unsqueeze(0)

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

        device = prompt_ids.device
        if response_chunks:
            full_response = torch.cat([c.to(device) for c in response_chunks], dim=-1)
        else:
            full_response = torch.zeros(1, 1, dtype=prompt_ids.dtype, device=device)

        full_ids = torch.cat([prompt_ids, full_response], dim=-1)
        seq_len = full_ids.shape[1]
        resp_len = full_response.shape[1]

        attn = torch.ones_like(full_ids)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)

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
