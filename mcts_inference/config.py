from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MCTSConfig:
    """Configuration for MCTS inference."""

    # Policy model
    policy_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    # MCTS tree
    max_tokens_per_node: int = 256       # each branch = 256 tokens
    max_depth: int = 16                  # max tree depth (~4096 tokens total)
    num_rollouts: int = 32               # MCTS iterations per problem
    num_children: int = 2                # children generated per expansion
    exploration_constant: float = 1.414  # UCB1 C parameter (sqrt(2))

    # Reward model
    prm_type: str = "logprob"            # "logprob" or "frozen_prm"
    prm_model_path: Optional[str] = None # path to trained PRM checkpoint
    prm_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    prm_value_head_type: str = "complex"
    prm_device: str = "cuda"

    # Stop strings are intentionally empty — we let each node generate
    # the full 256 tokens and detect terminal status post-hoc by checking
    # whether the cumulative reasoning contains a completed \boxed{...}.
    stop_strings: List[str] = field(default_factory=list)

    # System prompt
    system_prompt: str = (
        "You are a helpful assistant that solves math problems step by step. "
        "Show your reasoning clearly and put your final answer in \\boxed{}."
    )
