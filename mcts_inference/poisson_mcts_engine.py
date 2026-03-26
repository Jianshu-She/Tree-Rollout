"""Poisson-MCTS engine — distribution-guided tree search."""

from typing import Dict, Any, List

from config import MCTSConfig
from poisson_mcts_tree import PoissonMCTSTree
from policy_model import PolicyModel
from reward_model import build_reward_model
from utils import extract_answer


class PoissonMCTSEngine:

    def __init__(self, config: MCTSConfig, policy_model=None, reward_model=None):
        self.config = config
        if policy_model is not None:
            self.policy_model = policy_model
            self.reward_model = reward_model or build_reward_model(config)
        else:
            print("Initializing policy model ...")
            self.policy_model = PolicyModel(config)
            print("Initializing reward model ...")
            self.reward_model = build_reward_model(config)
        print("Poisson-MCTS engine ready.")

    def solve(self, question: str) -> Dict[str, Any]:
        tree = PoissonMCTSTree(question, self.config)

        for _ in range(self.config.num_rollouts):
            leaf = tree.select()

            if leaf.is_terminal:
                tree.backpropagate(leaf, leaf.prm_score)
                continue
            if leaf.depth >= self.config.max_depth:
                leaf.is_terminal = True
                tree.backpropagate(leaf, leaf.prm_score)
                continue

            children = tree.expand(leaf, self.policy_model, self.reward_model)
            for child in children:
                tree.backpropagate(child, child.prm_score)

        best_node = tree.best_terminal_node()
        all_terminals = tree.all_terminal_nodes()

        result: Dict[str, Any] = {
            "question": question,
            "tree_stats": tree.stats(),
            "tree": tree.to_dict(),
            "method": "poisson_mcts",
        }

        if best_node:
            solution = best_node.get_full_reasoning()
            result["best_solution"] = solution
            result["best_answer"] = extract_answer(solution)
            result["best_node_visits"] = best_node.visit_count
            result["best_node_q"] = best_node.q_value
            result["best_node_depth"] = best_node.depth

        result["all_solutions"] = []
        for t_node in sorted(all_terminals, key=lambda n: n.visit_count, reverse=True):
            sol_text = t_node.get_full_reasoning()
            result["all_solutions"].append({
                "answer": extract_answer(sol_text),
                "visits": t_node.visit_count,
                "q_value": t_node.q_value,
                "depth": t_node.depth,
            })

        return result
