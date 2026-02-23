import time
from typing import List, Dict, Any

from tqdm import tqdm

from config import MCTSConfig
from mcts_tree import MCTSTree
from policy_model import PolicyModel
from reward_model import build_reward_model, RewardModel
from utils import extract_answer, is_correct


class MCTSEngine:
    """Orchestrates MCTS inference for reasoning problems."""

    def __init__(self, config: MCTSConfig):
        self.config = config
        print("Initializing policy model ...")
        self.policy_model = PolicyModel(config)
        print("Initializing reward model ...")
        self.reward_model = build_reward_model(config)
        print("MCTS engine ready.")

    # ------------------------------------------------------------------
    # Solve a single problem
    # ------------------------------------------------------------------

    def solve(self, question: str) -> Dict[str, Any]:
        """Run MCTS on a single question and return the best solution."""
        tree = MCTSTree(question, self.config)

        for rollout_idx in range(self.config.num_rollouts):
            # 1. SELECT — walk to expandable leaf
            leaf = tree.select()

            # If the selected leaf is terminal, just backprop its score
            if leaf.is_terminal:
                tree.backpropagate(leaf, leaf.prm_score)
                continue

            # Skip if max depth exceeded
            if leaf.depth >= self.config.max_depth:
                leaf.is_terminal = True
                tree.backpropagate(leaf, leaf.prm_score)
                continue

            # 2. EXPAND — generate children
            children = tree.expand(leaf, self.policy_model, self.reward_model)

            # 3. BACKPROP — propagate each child's PRM score
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

        # All terminal solutions for analysis
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

    # ------------------------------------------------------------------
    # Batch solve
    # ------------------------------------------------------------------

    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run MCTS on a list of problems."""
        results = []
        pbar = tqdm(problems, desc="MCTS problems", unit="prob")

        for idx, problem in enumerate(pbar):
            question = problem["problem"]
            ground_truth = problem.get("answer", "")

            t0 = time.time()
            result = self.solve(question)
            elapsed = time.time() - t0

            result["problem_id"] = idx
            result["ground_truth"] = ground_truth
            result["elapsed_seconds"] = round(elapsed, 2)

            # Evaluate correctness
            if ground_truth and result.get("best_answer"):
                result["correct"] = is_correct(
                    result["best_solution"], ground_truth
                )
            else:
                result["correct"] = None

            results.append(result)

            # Progress update
            solved = sum(1 for r in results if r.get("correct") is True)
            total_eval = sum(1 for r in results if r.get("correct") is not None)
            acc = solved / total_eval if total_eval > 0 else 0
            pbar.set_postfix(
                acc=f"{acc:.1%}",
                nodes=result["tree_stats"]["total_nodes"],
                time=f"{elapsed:.1f}s",
            )

        return results
