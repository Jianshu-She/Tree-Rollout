# vt_ts_inference.py
"""Inference‑time utilities for Value‑Threshold Tree Search (VT‑TS).

This file **does not** re‑write the whole REBASE engine; it only provides:
1.  `load_value_model()` – load the trained value head weights.
2.  `vt_ts_select_and_expand()` – drop‑in replacement for Tree.select_and_expand
    that applies the dual‑threshold rule.

Integrate in three steps:
------------------------
1.  Place this file somewhere importable (e.g. project root).
2.  In your original REBASE search script, insert

    ```python
    from vt_ts_inference import load_value_model, vt_ts_select_and_expand

    val_model = load_value_model(
        base_model_name="Qwen/Qwen1.5-7B-Chat",          # same base
        val_ckpt="checkpoints/qwen_valhead.safetensors"
    ).cuda().eval()
    τ = 0.3   # original reward threshold
    τ2 = 0.55 # value‑head threshold
    ```
3.  Replace the line

    ```python
    continue_search = tree.select_and_expand(depth)
    ```
    with
    ```python
    continue_search = vt_ts_select_and_expand(tree, depth, val_model, τ, τ2)
    ```

No other changes needed.
"""

import torch, math
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SIG = torch.sigmoid

def load_value_model(base_model_name: str, val_ckpt: str, dtype=torch.bfloat16):
    """Load base LM *and* value‑head state dict (saved by train_value_head.py)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    wrapper_state = torch.load(val_ckpt, map_location="cpu")["model_state_dict"]
    # The state dict has keys: 'lm.*' (optional LoRA) and 'value_head.*'
    base.load_state_dict({k.replace("lm.", ""): v for k, v in wrapper_state.items() if k.startswith("lm.")}, strict=False)
    value_head = torch.nn.Linear(base.config.hidden_size, 1, bias=False)
    value_head.load_state_dict({"weight": wrapper_state["value_head.weight"]})

    class _Wrapper(torch.nn.Module):
        def __init__(self, lm, vh, tok):
            super().__init__()
            self.lm, self.vh, self.tokenizer = lm, vh, tok
        @torch.no_grad()
        def score_prefix(self, ids: torch.Tensor, mask: torch.Tensor):
            outs = self.lm(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            last = outs.hidden_states[-1]
            idx = mask.sum(-1) - 1
            h_L = last[torch.arange(last.size(0), device=last.device), idx]
            return self.vh(h_L).squeeze(-1)  # logits
    return _Wrapper(base, value_head, tokenizer)

# ---------------------------------------------------------------------------
# Dual‑threshold expansion ----------------------------------------------------
# ---------------------------------------------------------------------------

def _batch_value_scores(nodes, val_model):
    """Helper: compute value logits for a list of TreeNode."""
    ids, masks = [], []
    pad = val_model.tokenizer.pad_token_id
    for n in nodes:
        t = torch.tensor(n.get_state().ids(), dtype=torch.long)
        ids.append(t)
        masks.append(torch.ones_like(t))
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad).cuda()
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0).cuda()
    return val_model.score_prefix(ids, masks)


def vt_ts_select_and_expand(tree, depth: int, val_model, τ: float, τ2: float):
    """Drop‑in replacement for Tree.select_and_expand.

    Args:
        tree  : original Tree object.
        depth : which depth level to process.
        val_model : loaded via `load_value_model`.
        τ     : reward threshold (same as REBASE).
        τ2    : value‑head probability threshold.
    Returns:
        bool  : whether we expanded at least one node in this depth.
    """
    cand, scores = [], []
    # 1) gather candidate nodes that still have budget
    for node in tree.depth_nodes[depth]:
        if node.is_leaf() or node.get_cum_tokens() >= tree.paras["max_tokens"]:
            tree.remaining_width -= 1
        elif node.get_score() >= τ:
            cand.append(node)
            scores.append(node.get_score())
    if tree.remaining_width <= 0 or not cand:
        return False

    # 2) value‑head filter
    with torch.no_grad():
        v_logits = _batch_value_scores(cand, val_model)
    filt = [n for n, v in zip(cand, v_logits) if SIG(v) >= τ2]
    if not filt:
        return False

    # 3) allocate width (greedy: 1 per node) and expand
    alloc = min(len(filt), tree.remaining_width)
    for node in filt[:alloc]:
        tree.expand(node, 1)
    tree.remaining_width -= alloc
    return True
