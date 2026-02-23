# Tree-Rollout

Monte Carlo Tree Search (MCTS) for LLM mathematical reasoning. The system grows a search tree where each node is a 256-token reasoning chunk, using UCB1 selection, policy-model expansion, and process reward model (PRM) scoring to find high-quality solutions.

## Architecture

```
Question
    │
    ▼
┌────────┐   SELECT (UCB1)    ┌──────────┐   EXPAND (vLLM)    ┌──────────┐
│  Root  │ ─────────────────► │  Leaf    │ ─────────────────► │ Children │
│  Node  │                    │  Node    │                    │  Nodes   │
└────────┘                    └──────────┘                    └──────────┘
    ▲                                                              │
    │                        BACKPROPAGATE                         │
    └──────────────────────── (PRM score) ◄────────────────────────┘
```

Each MCTS iteration: **Select** a leaf via UCB1 → **Expand** it by generating child chunks with the policy LLM → **Score** children with the reward model → **Backpropagate** scores to root. After all rollouts, the terminal node with the highest visit count is chosen as the solution.

## Project Structure

```
mcts_inference/
├── config.py            # MCTSConfig dataclass (model, tree, reward settings)
├── mcts_node.py         # MCTSNode — single tree node with UCB1, path helpers
├── mcts_tree.py         # MCTSTree — select/expand/backprop + serialization
├── mcts_engine.py       # MCTSEngine — orchestrates solving, batch processing
├── policy_model.py      # vLLM-backed generation of 256-token chunks
├── reward_model.py      # Reward models: LogprobRewardModel, FrozenPRMRewardModel
├── utils.py             # Terminal detection, answer extraction, correctness check
├── visualize_tree.py    # Graphviz tree rendering (PNG/SVG) from results JSON
└── run_mcts.py          # CLI entry point
```

## Quick Start

### Run MCTS inference

```bash
cd mcts_inference

python run_mcts.py \
    --data ../data-prepare/data/MATH500_train.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_problems 5 \
    --num_rollouts 32 \
    --num_children 2 \
    --max_depth 16 \
    --tensor_parallel_size 4
```

Results are saved to `mcts_results/mcts_results_<timestamp>.json`.

### Visualize a search tree

```bash
python visualize_tree.py \
    --results mcts_results/mcts_results_XXXXXXXX_XXXXXX.json \
    --problem_id 0 \
    --output tree_p0.png
```

Supports `--format png`, `svg`, or `pdf`. Node colors:
- **Gray** — root node
- **Green** — terminal node (contains a completed `\boxed{...}` answer)
- **Orange** — non-terminal leaf (unexpanded)
- **Light blue** — internal node

The **best path** (root to highest-visit terminal) is highlighted with bold red edges.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--num_rollouts` | 32 | MCTS iterations per problem |
| `--num_children` | 2 | Children generated per node expansion |
| `--max_depth` | 16 | Maximum tree depth (~4096 tokens total) |
| `--max_tokens_per_node` | 256 | Tokens per reasoning chunk |
| `--exploration_constant` | 1.414 | UCB1 exploration parameter (sqrt(2)) |
| `--temperature` | 0.7 | Sampling temperature |
| `--prm_type` | `logprob` | Reward model: `logprob` (no extra model) or `frozen_prm` (trained PRM) |

## Reward Models

- **`logprob`** (default): Uses mean log-probability of generated tokens as a proxy reward, passed through a sigmoid. No additional model needed.
- **`frozen_prm`**: Uses a trained Process Reward Model that scores each reasoning step. Requires `--prm_model_path` pointing to a checkpoint.

## Dependencies

- [vLLM](https://github.com/vllm-project/vllm) — fast LLM inference
- [graphviz](https://pypi.org/project/graphviz/) (Python package + system binary) — tree visualization
- PyTorch, Transformers, NumPy
