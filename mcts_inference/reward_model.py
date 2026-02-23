import sys
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional

from config import MCTSConfig


class RewardModel(ABC):
    """Base class for reward models that score reasoning chunks."""

    @abstractmethod
    def score(self, question: str, reasoning_chunks: List[str]) -> List[float]:
        """Score each reasoning chunk. Returns list of scores in [0, 1]."""
        ...


# ------------------------------------------------------------------
# Logprob-based reward (default fallback)
# ------------------------------------------------------------------

class LogprobRewardModel(RewardModel):
    """Use mean log-probability as a proxy reward signal.

    Higher logprob → the model is more confident → higher reward.
    This requires no trained PRM checkpoint.
    """

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def score_from_logprobs(self, logprobs: List[float]) -> float:
        """Convert a list of per-token logprobs to a [0, 1] reward."""
        if not logprobs:
            return 0.5
        mean_lp = float(np.mean(logprobs))
        # sigmoid(mean_logprob * scale) → [0, 1]
        return 1.0 / (1.0 + np.exp(-mean_lp * self.scale))

    def score(self, question: str, reasoning_chunks: List[str]) -> List[float]:
        """Not directly usable without logprobs — see score_from_logprobs."""
        # This method exists for interface compatibility.
        # In practice the engine calls score_from_logprobs with the actual logprob data.
        return [0.5] * len(reasoning_chunks)


# ------------------------------------------------------------------
# Frozen PRM reward (when trained checkpoint is available)
# ------------------------------------------------------------------

class FrozenPRMRewardModel(RewardModel):
    """Wraps the ProcessRewardModelFrozen from prh/model_frozen.py."""

    def __init__(self, config: MCTSConfig):
        # Add prh/ to path so we can import the model
        prh_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "prh")
        if prh_dir not in sys.path:
            sys.path.insert(0, prh_dir)

        from model_frozen import ProcessRewardModelFrozen
        from transformers import AutoTokenizer

        self.device = config.prm_device
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.prm_base_model, trust_remote_code=True
        )

        if config.prm_model_path:
            self.model = ProcessRewardModelFrozen.from_pretrained(
                config.prm_model_path,
                base_model_name=config.prm_base_model,
                value_head_type=config.prm_value_head_type,
            )
        else:
            # Untrained PRM (random value head) — useful for structural testing
            self.model = ProcessRewardModelFrozen(
                model_name=config.prm_base_model,
                value_head_type=config.prm_value_head_type,
            )

        if self.device != "auto":
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, question: str, reasoning_chunks: List[str]) -> List[float]:
        """Score each reasoning chunk using the frozen PRM.

        Wraps chunks with <step>...</step> tags as expected by the PRM.
        """
        # Build input text
        parts = [question]
        for chunk in reasoning_chunks:
            parts.append(f"<step>{chunk}</step>")
        full_text = " ".join(parts)

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Find <step> token positions
        step_token_id = self.tokenizer.convert_tokens_to_ids("<step>")
        positions = []
        for i, tid in enumerate(inputs["input_ids"][0]):
            if tid.item() == step_token_id:
                positions.append(i)
        positions = positions[: len(reasoning_chunks)]

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            step_positions=[positions] if positions else None,
        )

        values = outputs["values"]
        if values.dim() == 0:
            scores = [torch.sigmoid(values).item()]
        elif values.dim() == 1:
            scores = torch.sigmoid(values).cpu().tolist()
        else:
            scores = torch.sigmoid(values[0]).cpu().tolist()

        # Pad / truncate to match chunk count
        scores = scores[: len(reasoning_chunks)]
        while len(scores) < len(reasoning_chunks):
            scores.append(0.5)
        return scores


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_reward_model(config: MCTSConfig) -> RewardModel:
    if config.prm_type == "frozen_prm":
        return FrozenPRMRewardModel(config)
    return LogprobRewardModel()
