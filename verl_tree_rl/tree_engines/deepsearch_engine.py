"""DeepSearch engine for verl GRPO training (arxiv 2509.25454).

Adapts DeepSearch MCTS (global frontier selection + entropy-guided scoring)
to verl's DataProto interface. Paper defaults (from github.com/smiles724/DeepSearch):
    global_lambda1 (quality)   = 0.4
    global_lambda2 (entropy)   = 0.4
    global_lambda3 (depth)     = 0.01
    depth_bonus_type           = sqrt
    expansion_width            = 8
    max_depth                  = 64
"""

import math
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from verl import DataProto

from verl_tree_rl.tree_engines.bfs_engine import _slice_dp, _pad_dps, _stack_dps, _Node

logger = logging.getLogger(__name__)


class DeepSearchEngine:
    """DeepSearch MCTS with global frontier selection + entropy guidance."""

    def __init__(self, config: Dict[str, Any], tokenizer=None):
        self.tokens_per_step = int(config.get("tokens_per_step", 256))
        self.max_depth = int(config.get("max_depth", 64))  # paper default
        self.expansion_width = int(config.get("expansion_width", 8))
        self.target_terminals = int(config.get("target_terminals", 32))
        self.max_rollouts = int(config.get("max_rollouts", 64))

        self.lambda_quality = float(config.get("lambda_quality", 0.4))
        self.lambda_entropy = float(config.get("lambda_entropy", 0.4))
        self.lambda_depth = float(config.get("lambda_depth", 0.01))
        self.depth_bonus_type = config.get("depth_bonus_type", "sqrt")

        logger.info(
            f"[DeepSearchEngine] lambdas=({self.lambda_quality},{self.lambda_entropy},"
            f"{self.lambda_depth}) depth_bonus={self.depth_bonus_type} "
            f"width={self.expansion_width} max_depth={self.max_depth}"
        )

    def _global_score(self, node: _Node, max_depth: int) -> float:
        """DeepSearch's F(s) global frontier priority score."""
        # Quality: parent Q (higher = more promising subtree)
        parent_q = getattr(node.parent, "q_value", 0.0) if node.parent else 0.0
        quality = self.lambda_quality * math.tanh(parent_q)

        # Entropy: token-level entropy proxy from this node's generation
        entropy = getattr(node, "entropy", 0.0)
        uncertainty = self.lambda_entropy * entropy

        # Depth bonus
        if self.depth_bonus_type == "sqrt":
            depth_bonus = self.lambda_depth * math.sqrt(
                node.depth / max(1, max_depth)
            )
        elif self.depth_bonus_type == "log":
            depth_bonus = self.lambda_depth * math.log(node.depth + 1)
        else:
            depth_bonus = self.lambda_depth * node.depth

        return quality + uncertainty + depth_bonus

    def _compute_entropy(self, child_dp: DataProto) -> float:
        """Approximate token-level entropy from rollout_log_probs if available."""
        logprobs = child_dp.batch.get("rollout_log_probs")
        if logprobs is None:
            return 0.0
        attn = child_dp.batch.get("attention_mask")
        if attn is not None:
            resp_len = logprobs.shape[-1]
            resp_attn = attn[0, -resp_len:].bool()
            valid = logprobs[0][resp_attn]
        else:
            valid = logprobs[0]
        if valid.numel() == 0:
            return 0.0
        return float(-valid.mean().item())

    def _get_response_tokens(self, node: _Node, device) -> torch.Tensor:
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

    def _assemble_prompt(self, original_prompt: DataProto, node: _Node) -> DataProto:
        device = original_prompt.batch["input_ids"].device
        prompt_ids = original_prompt.batch["input_ids"]
        deltas = node.full_trajectory_deltas()
        chunks = []
        for d in deltas:
            resp = d.batch.get("responses")
            if resp is None:
                continue
            attn = d.batch.get("attention_mask")
            if attn is not None:
                resp_attn = attn[0, -resp.shape[1]:]
                valid = resp_attn.bool()
                chunks.append(resp[0][valid].unsqueeze(0).to(device))
            else:
                chunks.append(resp.to(device))
        if chunks:
            new_ids = torch.cat([prompt_ids, torch.cat(chunks, dim=-1)], dim=-1)
        else:
            new_ids = prompt_ids
        seq_len = new_ids.shape[1]
        d = {
            "input_ids": new_ids,
            "attention_mask": torch.ones_like(new_ids),
            "position_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        }
        for k, v in original_prompt.non_tensor_batch.items():
            d[k] = v
        return DataProto.from_single_dict(d, meta_info=original_prompt.meta_info)

    def run(
        self,
        prompts: DataProto,
        inner_rollout,
        original_response_length: int,
        pad_token_id: int,
        n_per_prompt: int,
    ) -> DataProto:
        if pad_token_id is None:
            pad_token_id = 0
        pad_token_id = int(pad_token_id)
        bsz = prompts.batch["input_ids"].shape[0]
        prompt_len = prompts.batch["input_ids"].shape[1]
        resp_len = original_response_length
        device = prompts.batch["input_ids"].device

        all_prompts, all_responses, all_input_ids = [], [], []
        all_attn, all_pos = [], []
        all_ntb = {k: [] for k in prompts.non_tensor_batch.keys()}

        for i in range(bsz):
            single_prompt = _slice_dp(prompts, i, i + 1)
            terminal_nodes = self._build_tree_nodes(
                single_prompt, inner_rollout, pad_token_id
            )

            if len(terminal_nodes) > n_per_prompt:
                terminal_nodes = terminal_nodes[:n_per_prompt]
            while len(terminal_nodes) < n_per_prompt:
                terminal_nodes.append(
                    terminal_nodes[-1] if terminal_nodes else _Node("empty", 0, None)
                )

            prompt_ids = single_prompt.batch["input_ids"]

            for node in terminal_nodes:
                resp_tokens = self._get_response_tokens(node, device)
                if resp_tokens.shape[1] > resp_len:
                    resp_tokens = resp_tokens[:, :resp_len]
                elif resp_tokens.shape[1] < resp_len:
                    pad = torch.full(
                        (1, resp_len - resp_tokens.shape[1]), pad_token_id,
                        dtype=resp_tokens.dtype, device=device,
                    )
                    resp_tokens = torch.cat([resp_tokens, pad], dim=-1)

                full_ids = torch.cat([prompt_ids, resp_tokens], dim=-1)
                resp_real_len = min(
                    self._get_response_tokens(node, device).shape[1], resp_len
                )
                attn_resp = torch.cat([
                    torch.ones(1, resp_real_len, dtype=torch.long, device=device),
                    torch.zeros(1, resp_len - resp_real_len, dtype=torch.long, device=device),
                ], dim=-1)
                attn = torch.cat([single_prompt.batch["attention_mask"], attn_resp], dim=-1)
                prompt_pos = single_prompt.batch["position_ids"]
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

    def _build_tree_nodes(
        self, single_prompt: DataProto, inner_rollout, pad_token_id: int
    ) -> List[_Node]:
        """DeepSearch's global frontier selection loop."""
        uid = single_prompt.non_tensor_batch.get("uid", np.array(["0"]))[0]
        root = _Node(uid=str(uid), depth=0, delta=None)
        root.q_value = 0.0
        root.entropy = 0.0
        all_nodes = [root]
        terminals: List[_Node] = []

        print(f"[DeepSearch] start tree uid={uid} target={self.target_terminals}", flush=True)

        for rollout_idx in range(self.max_rollouts):
            if len(terminals) >= self.target_terminals:
                break

            # Global frontier: all non-terminal nodes
            frontier = [n for n in all_nodes if not n.is_terminal]
            if not frontier:
                break

            # Select node with highest F(s)
            best = max(frontier, key=lambda n: self._global_score(n, self.max_depth))

            # Expand with expansion_width children
            prompt_dp = self._assemble_prompt(single_prompt, best)
            batch_prompts = [deepcopy(prompt_dp) for _ in range(self.expansion_width)]
            batch_prompts = _pad_dps(batch_prompts, pad_token_id)
            gen_input = _stack_dps(batch_prompts)

            saved_max = getattr(inner_rollout.sampling_params, "max_tokens", None)
            try:
                inner_rollout.sampling_params.max_tokens = self.tokens_per_step
                outputs = inner_rollout.generate_sequences(gen_input)
            finally:
                if saved_max is not None:
                    inner_rollout.sampling_params.max_tokens = saved_max

            # Create children
            for j in range(self.expansion_width):
                child_dp = _slice_dp(outputs, j, j + 1)
                resp = child_dp.batch.get("responses")
                has_eos = False
                if resp is not None:
                    attn = child_dp.batch.get("attention_mask")
                    if attn is not None and attn.shape[-1] > 0:
                        resp_attn = attn[0, -resp.shape[1]:]
                        valid_tokens = int(resp_attn.sum().item())
                        has_eos = valid_tokens < self.tokens_per_step

                child = _Node(
                    uid=f"{best.uid}_r{rollout_idx}_c{j}",
                    depth=best.depth + 1,
                    delta=child_dp,
                    parent=best,
                )
                child.q_value = 0.0
                child.entropy = self._compute_entropy(child_dp)
                child.is_terminal = has_eos or (child.depth >= self.max_depth)
                best.children.append(child)
                all_nodes.append(child)

                if child.is_terminal:
                    terminals.append(child)

        print(f"[DeepSearch] tree done: rollouts={rollout_idx+1} "
              f"terminals={len(terminals)} nodes={len(all_nodes)}", flush=True)

        # Force remaining non-terminals at max depth
        for n in all_nodes:
            if not n.is_terminal and n.depth >= self.max_depth:
                n.is_terminal = True
                terminals.append(n)

        if not terminals:
            logger.warning("DeepSearch produced no terminals; returning root")
            terminals = [root]

        print(f"[DeepSearch] returning {len(terminals)} terminal nodes", flush=True)
        return terminals
