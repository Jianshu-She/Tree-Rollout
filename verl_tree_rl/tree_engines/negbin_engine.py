"""NegBin MCTS engine for verl GRPO training.

Adapts NegBin MCTS (from mcts_inference/poisson_mcts_engine.py) to verl's
DataProto interface. Key differences from BFS:
- Uses UCB1 selection to choose which node to expand
- Branching factor at each depth is SAMPLED from fitted NegBin/Poisson
  distributions (not deterministic mean)
- Has explicit target_terminals budget (like the offline engine)
- alpha parameter blends priority-weighted selection with fitted sampling

Reuses BFS engine's DataProto packing logic (assemble full trajectory
with prompt_ids + concatenated response chunks, pad to response_length).
"""

import json
import math
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from verl import DataProto

from verl_tree_rl.tree_engines.bfs_engine import _slice_dp, _pad_dps, _stack_dps, _Node

logger = logging.getLogger(__name__)


class NegBinMCTSEngine:
    """NegBin MCTS tree rollout for verl.

    Config keys (engine_kwargs):
        fitted_params_path: str   path to fitted_parameters.json
        training_stage: str       "step_0" / "step_40" / ...
        tokens_per_step: int      max_new_tokens per expansion (default 256)
        max_depth: int            max tree depth (default 12)
        target_terminals: int     minimum terminals per tree before stop (default 32)
        max_rollouts: int         hard cap on # of expansion rounds (default 64)
        alpha: float              UCB exploration coefficient (default 0.5)
        exploration_constant: float  UCB1 c (default 1.414)
    """

    def __init__(self, config: Dict[str, Any], tokenizer=None):
        self.tokens_per_step = int(config.get("tokens_per_step", 256))
        self.max_depth = int(config.get("max_depth", 12))
        self.target_terminals = int(config.get("target_terminals", 32))
        self.max_rollouts = int(config.get("max_rollouts", 64))
        self.alpha = float(config.get("alpha", 0.5))
        self.c = float(config.get("exploration_constant", 1.414))

        # Load fitted parameters
        params_path = config.get("fitted_params_path", "")
        stage = config.get("training_stage", "step_0")
        self.bf_samplers = {}  # depth -> callable returning bf ~ fitted distribution
        self.depth_mean_bf = {}
        if params_path:
            with open(params_path) as f:
                all_params = json.load(f)
            stage_params = all_params.get(stage, {})
            for d_str, entry in stage_params.items():
                depth = int(d_str)
                # D0: NegBin, D1+: Poisson (per paper). Approximate with sampling.
                if "negbin_r" in entry and "negbin_p" in entry and depth == 0:
                    r = entry["negbin_r"]
                    p = entry["negbin_p"]
                    self.bf_samplers[depth] = lambda r=r, p=p: max(
                        1, int(np.random.negative_binomial(r, p))
                    )
                    self.depth_mean_bf[depth] = entry.get("bf_mean", 1.0)
                elif "poisson_lambda" in entry:
                    lam = entry["poisson_lambda"]
                    self.bf_samplers[depth] = lambda lam=lam: max(
                        1, int(np.random.poisson(lam))
                    )
                    self.depth_mean_bf[depth] = lam
        else:
            logger.warning("No fitted_params_path; using default bf sampler [8,2,1]")
            self.bf_samplers = {
                0: lambda: 8,
                1: lambda: 2,
                2: lambda: 1,
            }
            self.depth_mean_bf = {0: 8.0, 1: 2.0, 2: 1.0}

        logger.info(
            f"[NegBinMCTSEngine] alpha={self.alpha} c={self.c} "
            f"max_depth={self.max_depth} target={self.target_terminals} "
            f"depth_mean_bf={self.depth_mean_bf}"
        )

    def sample_bf(self, depth: int) -> int:
        if depth in self.bf_samplers:
            return self.bf_samplers[depth]()
        return 1

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

    def run(
        self,
        prompts: DataProto,
        inner_rollout,
        original_response_length: int,
        pad_token_id: int,
        n_per_prompt: int,
    ) -> DataProto:
        """Run NegBin MCTS. Returns DataProto with fixed shapes."""
        if pad_token_id is None:
            pad_token_id = 0
        pad_token_id = int(pad_token_id)
        bsz = prompts.batch["input_ids"].shape[0]
        prompt_len = prompts.batch["input_ids"].shape[1]
        resp_len = original_response_length
        total_len = prompt_len + resp_len
        device = prompts.batch["input_ids"].device

        all_prompts, all_responses, all_input_ids = [], [], []
        all_attn, all_pos = [], []
        all_ntb = {k: [] for k in prompts.non_tensor_batch.keys()}

        # BATCHED: MCTS rollouts run across all trees in parallel.
        # At each iteration: select best UCB node per tree → batch expand with
        # sampled bf children → backprop Q values.
        forest_terminals = self._build_forest_batched_mcts(
            prompts, inner_rollout, pad_token_id
        )

        for i in range(bsz):
            single_prompt = _slice_dp(prompts, i, i + 1)
            terminal_nodes = forest_terminals[i]

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
                        dtype=resp_tokens.dtype, device=device
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

    def _ucb1(self, node: _Node, total_visits: int) -> float:
        """UCB1 score for node selection."""
        visits = max(1, getattr(node, "visit_count", 0))
        q = getattr(node, "q_value", 0.0)
        explore = self.c * math.sqrt(math.log(max(total_visits, 2)) / visits)
        return q + self.alpha * explore

    def _build_forest_batched_mcts(
        self,
        prompts: DataProto,
        inner_rollout,
        pad_token_id: int,
    ) -> List[List[_Node]]:
        """Batched MCTS across all prompts. At each iteration:
        - Select best UCB1 non-terminal node per tree (top-1)
        - Sample bf at that depth from fitted NegBin/Poisson
        - Batch-expand all selected nodes in one generate_sequences call
        - Create children, mark terminals, update visit counts

        Returns per_tree_terminals[i] = list of terminal _Node for tree i.
        """
        bsz = prompts.batch["input_ids"].shape[0]
        roots: List[_Node] = []
        for i in range(bsz):
            uid_val = str(i)
            if "uid" in prompts.non_tensor_batch:
                uid_val = str(prompts.non_tensor_batch["uid"][i])
            root = _Node(uid=uid_val, depth=0, delta=None)
            root.visit_count = 1
            root.q_value = 0.0
            roots.append(root)

        all_nodes_per_tree: List[List[_Node]] = [[r] for r in roots]
        per_tree_terminals: List[List[_Node]] = [[] for _ in range(bsz)]

        print(f"[NegBin] batched forest: bsz={bsz} target={self.target_terminals} "
              f"max_rollouts={self.max_rollouts}", flush=True)

        for rollout_idx in range(self.max_rollouts):
            # Stop early if all trees have enough terminals
            if all(len(t) >= self.target_terminals for t in per_tree_terminals):
                print(f"[NegBin] all trees hit target at rollout={rollout_idx}",
                      flush=True)
                break

            # Select best non-terminal node per tree via UCB1
            selected_per_tree: List[Optional[_Node]] = []
            for tree_idx in range(bsz):
                if len(per_tree_terminals[tree_idx]) >= self.target_terminals:
                    selected_per_tree.append(None)  # skip this tree
                    continue
                candidates = [n for n in all_nodes_per_tree[tree_idx] if not n.is_terminal]
                if not candidates:
                    selected_per_tree.append(None)
                    continue
                total_visits = sum(max(1, n.visit_count) for n in all_nodes_per_tree[tree_idx])
                best = max(candidates, key=lambda n: self._ucb1(n, total_visits))
                selected_per_tree.append(best)

            # Build batch of prompts to expand
            all_gen_prompts: List[DataProto] = []
            all_parent_refs: List[tuple] = []  # (tree_idx, parent, bf)
            for tree_idx, parent in enumerate(selected_per_tree):
                if parent is None:
                    continue
                bf = self.sample_bf(parent.depth)
                original_prompt = _slice_dp(prompts, tree_idx, tree_idx + 1)
                prompt_dp = self._assemble_prompt(original_prompt, parent)
                for _ in range(bf):
                    all_gen_prompts.append(deepcopy(prompt_dp))
                    all_parent_refs.append((tree_idx, parent, bf))

            if not all_gen_prompts:
                print(f"[NegBin] no expandable nodes at rollout={rollout_idx}",
                      flush=True)
                break

            # Batched generation
            all_gen_prompts = _pad_dps(all_gen_prompts, pad_token_id)
            gen_input = _stack_dps(all_gen_prompts)
            saved_max = getattr(inner_rollout.sampling_params, "max_tokens", None)
            try:
                inner_rollout.sampling_params.max_tokens = self.tokens_per_step
                outputs = inner_rollout.generate_sequences(gen_input)
            finally:
                if saved_max is not None:
                    inner_rollout.sampling_params.max_tokens = saved_max

            # Distribute outputs back to trees
            for j, (tree_idx, parent, _bf) in enumerate(all_parent_refs):
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
                    uid=f"t{tree_idx}_r{rollout_idx}_j{j}",
                    depth=parent.depth + 1,
                    delta=child_dp,
                    parent=parent,
                )
                child.visit_count = 1
                child.q_value = 0.0
                child.is_terminal = has_eos or (child.depth >= self.max_depth)
                parent.children.append(child)
                all_nodes_per_tree[tree_idx].append(child)
                if child.is_terminal:
                    per_tree_terminals[tree_idx].append(child)

            # Update visit counts up the tree for each expanded parent (per tree, unique)
            seen = set()
            for tree_idx, parent, _bf in all_parent_refs:
                if (tree_idx, id(parent)) in seen:
                    continue
                seen.add((tree_idx, id(parent)))
                node = parent
                while node is not None:
                    node.visit_count += 1
                    node = node.parent

            if rollout_idx % 10 == 9:
                counts = [len(t) for t in per_tree_terminals]
                print(f"[NegBin] rollout={rollout_idx+1}: terminals_per_tree "
                      f"min={min(counts)} max={max(counts)} "
                      f"mean={sum(counts)/max(1,len(counts)):.1f}", flush=True)

        # Force remaining non-terminal nodes at max_depth as terminals
        for tree_idx in range(bsz):
            for n in all_nodes_per_tree[tree_idx]:
                if not n.is_terminal and n.depth >= self.max_depth:
                    n.is_terminal = True
                    per_tree_terminals[tree_idx].append(n)

        # Fallback for empty trees
        for tree_idx in range(bsz):
            if not per_tree_terminals[tree_idx]:
                per_tree_terminals[tree_idx] = [roots[tree_idx]]

        counts = [len(t) for t in per_tree_terminals]
        print(f"[NegBin] forest done: mean_terminals={sum(counts)/max(1,len(counts)):.1f}",
              flush=True)
        return per_tree_terminals

    def _build_tree_nodes(
        self, single_prompt: DataProto, inner_rollout, pad_token_id: int
    ) -> List[_Node]:
        """MCTS-style tree building. Simplified: repeated BFS-like expansion
        with UCB selection on non-terminal nodes."""
        uid = single_prompt.non_tensor_batch.get("uid", np.array(["0"]))[0]
        root = _Node(uid=str(uid), depth=0, delta=None)
        # Augment with MCTS fields
        root.visit_count = 1
        root.q_value = 0.0
        frontier = [root]
        all_nodes = [root]
        terminals: List[_Node] = []

        print(f"[NegBin] start tree uid={uid} target={self.target_terminals}", flush=True)

        for rollout_idx in range(self.max_rollouts):
            if len(terminals) >= self.target_terminals:
                break
            if not frontier:
                break

            # Select most promising non-terminal nodes (UCB1 over all non-terminal)
            candidates = [n for n in all_nodes if not n.is_terminal]
            if not candidates:
                break
            total_visits = sum(max(1, getattr(n, "visit_count", 0)) for n in all_nodes)
            # Score and pick top-k candidates to expand this round
            scored = sorted(
                candidates,
                key=lambda n: -self._ucb1(n, total_visits),
            )
            # Expand top 1-2 nodes per round (MCTS iterative)
            expand_nodes = scored[:1]

            # Sample branching factor for this expansion
            batch_prompts = []
            batch_nodes = []
            for parent in expand_nodes:
                bf = self.sample_bf(parent.depth)
                prompt_dp = self._assemble_prompt(single_prompt, parent)
                for _ in range(bf):
                    batch_prompts.append(deepcopy(prompt_dp))
                    batch_nodes.append(parent)

            if not batch_prompts:
                break

            # Pad prompts to same length and stack
            from verl_tree_rl.tree_engines.bfs_engine import _pad_dps, _stack_dps
            batch_prompts = _pad_dps(batch_prompts, pad_token_id)
            gen_input = _stack_dps(batch_prompts)

            # Generate with short chunks
            saved_max = getattr(inner_rollout.sampling_params, "max_tokens", None)
            try:
                inner_rollout.sampling_params.max_tokens = self.tokens_per_step
                outputs = inner_rollout.generate_sequences(gen_input)
            finally:
                if saved_max is not None:
                    inner_rollout.sampling_params.max_tokens = saved_max

            # Create children and backprop Q
            from verl_tree_rl.tree_engines.bfs_engine import _slice_dp as _slice
            for j, parent in enumerate(batch_nodes):
                child_dp = _slice(outputs, j, j + 1)
                resp = child_dp.batch.get("responses")
                has_eos = False
                if resp is not None:
                    attn = child_dp.batch.get("attention_mask")
                    if attn is not None and attn.shape[-1] > 0:
                        resp_attn = attn[0, -resp.shape[1]:]
                        valid_tokens = int(resp_attn.sum().item())
                        has_eos = valid_tokens < self.tokens_per_step

                child = _Node(
                    uid=f"{parent.uid}_r{rollout_idx}_c{j}",
                    depth=parent.depth + 1,
                    delta=child_dp,
                    parent=parent,
                )
                child.visit_count = 1
                child.q_value = 0.0
                child.is_terminal = has_eos or (child.depth >= self.max_depth)
                parent.children.append(child)
                all_nodes.append(child)

                if child.is_terminal:
                    terminals.append(child)

            # Update visit_count up the tree
            for parent in expand_nodes:
                node = parent
                while node is not None:
                    node.visit_count = getattr(node, "visit_count", 0) + 1
                    node = node.parent

        print(f"[NegBin] tree done: rollouts={rollout_idx+1} terminals={len(terminals)} "
              f"all_nodes={len(all_nodes)}", flush=True)

        # Force any non-terminal nodes at max depth as terminals
        for n in all_nodes:
            if not n.is_terminal and n.depth >= self.max_depth:
                n.is_terminal = True
                terminals.append(n)

        if not terminals:
            logger.warning("NegBin MCTS produced no terminals; returning root")
            terminals = [root]

        print(f"[NegBin] returning {len(terminals)} terminal nodes", flush=True)
        return terminals

    def _assemble_prompt(self, original_prompt: DataProto, node: _Node) -> DataProto:
        """Construct generation input = original prompt + accumulated chunks."""
        if node.depth == 0 and node.delta is None:
            return deepcopy(original_prompt)
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
            all_resp = torch.cat(chunks, dim=-1)
            new_ids = torch.cat([prompt_ids, all_resp], dim=-1)
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
