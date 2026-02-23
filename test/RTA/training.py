#!/usr/bin/env python3
"""train_value_head.py
------------------------------------------------------------
Trains a lightweight value head (linear probe or LoRA‑adapted) on top of a
causal language model, using <prefix, reward_final> pairs that you collected
from running the original REBASE search.

Input JSONL format (one line per sample):
{
  "prompt": "... original question ...",
  "prefix": [128, 297, 45, ...],          # token‑ids of the partial CoT chain
  "reward_final": 0 | 1                   # exact‑match outcome of the full chain
}

Usage (single‑GPU example)
-------------------------
python train_value_head.py \
    --base_model Qwen/Qwen1.5-7B-Chat \
    --train_json data/partial2final.jsonl \
    --output_path checkpoints/qwen_valhead.safetensors \
    --epochs 1 --bsz 64 --lr 5e-5 --lora_rank 4

The script saves **only** the value‑head & any LoRA layers in a `safetensors`
file (so it stays lightweight).  At inference time you load the base model
plus these weights with `peft`.
"""

import argparse, json, os, math, random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# ---------------------------------------------------------------------------
# Dataset --------------------------------------------------------------------
# ---------------------------------------------------------------------------
class PrefixRewardDataset(Dataset):
    """Loads <prefix, reward_final> samples.

    The `prefix` field should already be a list of token‑ids.  No extra
    tokenisation is done to keep this file format simple and fast.
    """

    def __init__(self, json_path: str, max_len: int = 1024):
        super().__init__()
        self.samples: List[Dict] = []
        with open(json_path) as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                ids: List[int] = j["prefix"][: max_len]
                # Convert list[int] -> tensor here to avoid realloc in __getitem__
                self.samples.append({
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "reward": torch.tensor(j["reward_final"], dtype=torch.float)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch, pad_id: int):
    """Pad input_ids to max length in batch; produce attention_mask."""
    lengths = [ex["input_ids"].size(0) for ex in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros_like(input_ids)
    rewards = torch.zeros(len(batch), dtype=torch.float)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].size(0)
        input_ids[i, :L] = ex["input_ids"]
        attn_mask[i, :L] = 1
        rewards[i] = ex["reward"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "reward": rewards}

# ---------------------------------------------------------------------------
# Model wrapper ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class ValueHeadWrapper(nn.Module):
    """Causal LM + linear value head (+ optional LoRA on FFN)."""
    def __init__(self, base_model_name: str, lora_rank: int | None = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        if lora_rank is not None and lora_rank > 0:
            lora_cfg = LoraConfig(r=lora_rank, lora_alpha=32, target_modules=["up_proj","down_proj"], bias="none")
            model = get_peft_model(model, lora_cfg)
        self.lm = model
        self.value_head = nn.Linear(model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        last_hid = outs.hidden_states[-1]
        # gather last token hidden per sample
        idx = attention_mask.sum(-1) - 1  # [B]
        last_vec = last_hid[torch.arange(last_hid.size(0), device=last_hid.device), idx]
        v_logit = self.value_head(last_vec).squeeze(-1)
        return v_logit

# ---------------------------------------------------------------------------
# Training loop ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    accelerator = Accelerator(mixed_precision="bf16")
    torch.manual_seed(args.seed)

    model = ValueHeadWrapper(args.base_model, args.lora_rank)
    tokenizer = model.tokenizer
    dataset = PrefixRewardDataset(args.train_json, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, pin_memory=True,
                            collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))

    model, dataloader = accelerator.prepare(model, dataloader)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=len(dataloader)*args.epochs)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, batch["reward"])
            accelerator.backward(loss)
            optim.step(); optim.zero_grad(); sched.step()
            if accelerator.is_main_process and step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(dataloader)}  loss {loss.item():.4f}")

    if accelerator.is_main_process:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": accelerator.unwrap_model(model).state_dict()}, args.output_path)
        print(f"[Value‑Head] saved to {args.output_path}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="huggingface model name or local path")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(args)
