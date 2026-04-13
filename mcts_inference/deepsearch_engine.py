"""DeepSearch MCTS engine — uses global frontier selection + entropy guidance."""

import time
from typing import List, Dict, Any

from tqdm import tqdm

from config import MCTSConfig
from deepsearch_tree import DeepSearchTree
from policy_model import PolicyModel
from reward_model import build_reward_model, RewardModel
from utils import extract_answer, is_correct


class DeepSearchEngine:
    """Orchestrates DeepSearch-style MCTS inference."""

    def __init__(self, config: MCTSConfig, policy_model=None, reward_model=None):
        self.config = config
        if policy_model is None:
            print("Initializing policy model ...")
            policy_model = PolicyModel(config)
        if reward_model is None:
            print("Initializing reward model ...")
            reward_model = build_reward_model(config)
        self.policy_model = policy_model
        self.reward_model = reward_model
        print("DeepSearch engine ready.")

    def solve_to_target(
        self, question: str, target_terminals: int = 128, max_rollouts: int = 512
    ) -> "DeepSearchTree":
        """Run DeepSearch MCTS until at least target_terminals terminal nodes exist.

        Mirrors PoissonMCTSEngine.solve_to_target for parallel benchmarking.
        Returns the live tree object so callers can walk terminal nodes and
        grab full reasoning text.
        """
        tree = DeepSearchTree(question, self.config)

        for _ in range(max_rollouts):
            if len(tree.all_terminal_nodes()) >= target_terminals:
                break

            node = tree.global_select()

            if node.is_terminal:
                tree.backpropagate(node, node.prm_score)
                continue
            if node.depth >= self.config.max_depth:
                node.is_terminal = True
                tree.backpropagate(node, node.prm_score)
                continue

            children = tree.expand(node, self.policy_model, self.reward_model)
            for child in children:
                tree.backpropagate(child, child.prm_score)

        return tree

    def solve(self, question: str) -> Dict[str, Any]:
        """Run DeepSearch MCTS on a single question."""
        tree = DeepSearchTree(question, self.config)

        for rollout_idx in range(self.config.num_rollouts):
            # 1. GLOBAL SELECT — pick the best node to expand across entire tree
            node = tree.global_select()

            # If terminal, just backprop
            if node.is_terminal:
                tree.backpropagate(node, node.prm_score)
                continue

            if node.depth >= self.config.max_depth:
                node.is_terminal = True
                tree.backpropagate(node, node.prm_score)
                continue

            # 2. EXPAND — generate children (width depends on depth)
            children = tree.expand(node, self.policy_model, self.reward_model)

            # 3. BACKPROP
            for child in children:
                tree.backpropagate(child, child.prm_score)

        # Extract results
        best_node = tree.best_terminal_node()
        best_q_node = tree.best_q_terminal()
        all_terminals = tree.all_terminal_nodes()

        result: Dict[str, Any] = {
            "question": question,
            "tree_stats": tree.stats(),
            "tree": tree.to_dict(),
            "method": "deepsearch",
        }

        if best_node:
            solution = best_node.get_full_reasoning()
            result["best_solution"] = solution
            result["best_answer"] = extract_answer(solution)
            result["best_node_visits"] = best_node.visit_count
            result["best_node_q"] = best_node.q_value
            result["best_node_depth"] = best_node.depth

        if best_q_node and best_q_node is not best_node:
            q_solution = best_q_node.get_full_reasoning()
            result["best_q_solution"] = q_solution
            result["best_q_answer"] = extract_answer(q_solution)

        result["all_solutions"] = []
        for t_node in sorted(all_terminals, key=lambda n: n.visit_count, reverse=True):
            sol_text = t_node.get_full_reasoning()
            result["all_solutions"].append({
                "answer": extract_answer(sol_text),
                "visits": t_node.visit_count,
                "q_value": t_node.q_value,
                "prm_score": t_node.prm_score,
                "depth": t_node.depth,
                "total_tokens": t_node.get_total_tokens(),
                "text_preview": sol_text[:200],
            })

        return result
