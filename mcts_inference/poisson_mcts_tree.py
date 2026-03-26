"""Poisson-MCTS: Distribution-guided tree search.

Branching factor at each depth is sampled from distributions fitted
to flat rollout post-hoc trees:
  - D0: NegBin(r, p) — captures overdispersion
  - D1+: Poisson(λ_d) — fast decay toward 1

Selection blends UCB1 with Beta-prior noise controlled by α:
  - α=1: near-random selection (matches flat rollout independence)
  - α=0: pure UCB1 (greedy exploitation)

Structural KL divergence tracks how well the tree matches flat rollout stats.
"""

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from config import MCTSConfig
from mcts_node import MCTSNode
from policy_model import PolicyModel
from reward_model import RewardModel, LogprobRewardModel
from utils import is_terminal_text


class PoissonMCTSTree:

    def __init__(self, question: str, config: MCTSConfig):
        self.question = question
        self.config = config
        self.root = MCTSNode(text="", parent=None)
        self.root.visit_count = 1
        self.total_nodes = 1

        # Poisson-MCTS parameters (attached to config)
        self.alpha = getattr(config, "pm_alpha", 0.5)
        self.bf_temperature = getattr(config, "pm_bf_temperature", 1.0)
        self.fitted_params_path = getattr(
            config, "pm_fitted_params_path",
            "faithful_baseline/results/math500_full/train/poisson_beta_analysis/fitted_parameters.json",
        )
        self.training_stage = getattr(config, "pm_training_stage", "step_0")
        self.use_negbin_d0 = getattr(config, "pm_use_negbin_d0", True)
        self.max_bf_d0 = getattr(config, "pm_max_bf_d0", 32)
        self.max_bf_d1plus = getattr(config, "pm_max_bf_d1plus", 8)

        # Load distributions
        self._dist_params: Dict[int, dict] = {}
        self._load_distributions()

        # Track branching factors at each depth for KL computation
        self._actual_bf: Dict[int, List[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Distribution loading
    # ------------------------------------------------------------------

    def _load_distributions(self):
        with open(self.fitted_params_path) as f:
            all_params = json.load(f)
        stage_params = all_params.get(self.training_stage, {})
        for d_str, params in stage_params.items():
            self._dist_params[int(d_str)] = params

    # ------------------------------------------------------------------
    # Branching factor sampling
    # ------------------------------------------------------------------

    def get_expansion_width(self, depth: int) -> int:
        """Sample branching factor from the fitted distribution at this depth."""
        params = self._dist_params.get(depth, {})

        if depth == 0 and self.use_negbin_d0:
            r = params.get("negbin_r")
            p = params.get("negbin_p")
            if r is not None and p is not None:
                sample = int(sp_stats.nbinom.rvs(n=r, p=p))
                sample = self._apply_temperature(sample, r * (1 - p) / p)
                return max(1, min(self.max_bf_d0, sample))

        # Poisson for D0 (fallback) and D1+
        lam = params.get("poisson_lambda", 1.0)
        if lam < 1.01:
            return 1  # effectively no branching
        sample = int(sp_stats.poisson.rvs(mu=lam))
        sample = self._apply_temperature(sample, lam)
        max_bf = self.max_bf_d0 if depth == 0 else self.max_bf_d1plus
        return max(1, min(max_bf, sample))

    def _apply_temperature(self, sample: int, mean: float) -> int:
        """Apply temperature scaling: T<1 pulls sample toward mean."""
        if self.bf_temperature >= 1.0:
            return sample
        # Interpolate between mean and sample
        blended = mean + self.bf_temperature * (sample - mean)
        return max(1, int(round(blended)))

    # ------------------------------------------------------------------
    # Selection: hybrid UCB1 + Beta-prior noise
    # ------------------------------------------------------------------

    def select(self) -> MCTSNode:
        """Select a leaf node for expansion using hybrid UCB1/random."""
        node = self.root
        while node.children and not node.is_terminal:
            node = self._select_child(node)
        return node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Pick a child using blended UCB1 + Beta-prior noise.

        alpha=0 → pure UCB1 (exploitation-focused)
        alpha=1 → near-random (diversity, matching flat rollout independence)
        """
        C = self.config.exploration_constant
        best_child = None
        best_score = float("-inf")

        for child in node.children:
            # UCB1 component
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploit = child.q_value
                explore = C * math.sqrt(
                    math.log(node.visit_count + 1) / (child.visit_count + 1)
                )
                ucb = exploit + explore

            # Beta-prior noise component (adds diversity)
            depth = child.depth
            params = self._dist_params.get(depth, {})
            beta_a = params.get("beta_alpha", 0.5)
            beta_b = params.get("beta_beta", 0.5)
            noise = sp_stats.beta.rvs(beta_a, beta_b)

            # Blend: (1-α) * UCB1 + α * noise
            score = (1.0 - self.alpha) * ucb + self.alpha * noise

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(
        self,
        node: MCTSNode,
        policy_model: PolicyModel,
        reward_model: RewardModel,
    ) -> List[MCTSNode]:
        if node.is_terminal:
            return []
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return []

        n_children = self.get_expansion_width(node.depth)
        self._actual_bf[node.depth].append(n_children)

        partial = node.get_full_reasoning()
        continuations = policy_model.generate_continuations(
            question=self.question,
            partial_reasoning=partial,
            n=n_children,
        )

        new_children: List[MCTSNode] = []
        for text, logprobs, hit_stop in continuations:
            if not text.strip():
                continue
            full_reasoning = partial + text
            terminal = hit_stop or is_terminal_text(full_reasoning)

            if isinstance(reward_model, LogprobRewardModel):
                score = reward_model.score_from_logprobs(logprobs)
            else:
                chunks = node.get_path_texts()[1:]
                chunks.append(text)
                scores = reward_model.score(self.question, chunks)
                score = scores[-1]

            child = MCTSNode(
                text=text,
                parent=None,
                prm_score=score,
                token_count=len(logprobs) if logprobs else len(text.split()),
                is_terminal=terminal,
            )
            node.add_child(child)
            new_children.append(child)
            self.total_nodes += 1

        if not new_children and not node.children:
            node.is_terminal = True

        return new_children

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    @staticmethod
    def backpropagate(node: MCTSNode, value: float) -> None:
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Structural KL divergence
    # ------------------------------------------------------------------

    def compute_structural_kl(self) -> float:
        """Compute KL(actual || target) of branching factor distributions.

        Uses Poisson KL: KL(λ1 || λ2) = λ1 - λ2 + λ2*ln(λ2/λ1)
        Returns average across depths.
        """
        kl_sum = 0.0
        n_depths = 0

        for depth in range(self.config.max_depth):
            actual_bfs = self._actual_bf.get(depth, [])
            if not actual_bfs:
                continue
            lambda_actual = np.mean(actual_bfs)
            params = self._dist_params.get(depth, {})
            lambda_target = params.get("poisson_lambda", 1.0)

            if lambda_actual <= 0 or lambda_target <= 0:
                continue

            # Poisson KL divergence
            kl = lambda_actual - lambda_target + lambda_target * math.log(
                lambda_target / lambda_actual
            )
            kl_sum += abs(kl)  # symmetric-ish
            n_depths += 1

        return kl_sum / max(n_depths, 1)

    def get_branching_profile(self) -> Dict[int, float]:
        """Return actual average branching factor at each depth."""
        return {
            d: float(np.mean(bfs))
            for d, bfs in sorted(self._actual_bf.items())
        }

    def get_target_profile(self) -> Dict[int, float]:
        """Return target branching factor at each depth."""
        return {
            d: params.get("poisson_lambda", 1.0)
            for d, params in sorted(self._dist_params.items())
        }

    # ------------------------------------------------------------------
    # Best path extraction
    # ------------------------------------------------------------------

    def best_terminal_node(self) -> Optional[MCTSNode]:
        terminals = self._collect_terminals(self.root)
        if not terminals:
            return None
        return max(terminals, key=lambda n: n.visit_count)

    def all_terminal_nodes(self) -> List[MCTSNode]:
        return self._collect_terminals(self.root)

    def _collect_terminals(self, node: MCTSNode) -> List[MCTSNode]:
        if node.is_terminal:
            return [node]
        result = []
        for child in node.children:
            result.extend(self._collect_terminals(child))
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        counter = [0]

        def _ser(node: MCTSNode) -> dict:
            nid = counter[0]
            counter[0] += 1
            preview = node.text[:50].replace("\n", " ") if node.text else "(root)"
            return {
                "id": nid,
                "depth": node.depth,
                "text_preview": preview,
                "visit_count": node.visit_count,
                "q_value": round(node.q_value, 4),
                "prm_score": round(node.prm_score, 4),
                "is_terminal": node.is_terminal,
                "token_count": node.token_count,
                "children": [_ser(c) for c in node.children],
            }

        return _ser(self.root)

    def stats(self) -> dict:
        terminals = self.all_terminal_nodes()
        return {
            "total_nodes": self.total_nodes,
            "terminal_nodes": len(terminals),
            "max_depth_reached": max((t.depth for t in terminals), default=0),
            "root_visits": self.root.visit_count,
            "structural_kl": round(self.compute_structural_kl(), 6),
            "branching_profile": self.get_branching_profile(),
            "target_profile": {
                d: v for d, v in self.get_target_profile().items() if d <= 10
            },
        }
