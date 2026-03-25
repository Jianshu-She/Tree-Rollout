#!/usr/bin/env python3
"""Node-level accuracy analysis: build trees using two clustering methods
(syntactic substring matching vs semantic LLM clustering), then trace each
node's descendant leaves to compute correct/incorrect probability distributions.

Step 1: Build trees with syntactic matching (multiple thresholds) and semantic matching
Step 2: For each node, compute accuracy distribution from its leaf descendants
Step 3: Visualize with scatter plots, histograms, and box plots
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcts_inference"))

from overlap_analysis.step_overlap_analysis import (
    chunk_text_into_steps,
    build_tree_for_problem,
    cluster_steps_heuristic,
)
from mcts_inference.utils import extract_answer, is_correct


# ============================================================================
# Semantic clustering: Embedding + LLM refinement (handles 128+ rollouts)
# ============================================================================

async def get_embeddings_batch(client, texts, model="text-embedding-3-small"):
    """Get embeddings for a list of texts using OpenAI API."""
    resp = await client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def cluster_by_embeddings(embeddings, threshold=0.85):
    """Greedy clustering based on cosine similarity of embeddings."""
    n = len(embeddings)
    emb_array = np.array(embeddings)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb_array / norms

    labels = [-1] * n
    centroids = []  # list of (centroid_embedding, cluster_id)
    next_cid = 0

    for i in range(n):
        assigned = False
        if centroids:
            centroid_matrix = np.array([c[0] for c in centroids])
            sims = emb_norm[i] @ centroid_matrix.T
            best_idx = np.argmax(sims)
            if sims[best_idx] >= threshold:
                labels[i] = centroids[best_idx][1]
                assigned = True

        if not assigned:
            labels[i] = next_cid
            centroids.append((emb_norm[i], next_cid))
            next_cid += 1

    return labels


async def _llm_judge_merge(client, semaphore, rep_texts, model="gpt-4o"):
    """Use a powerful LLM to judge which representative clusters should be merged.

    Takes representatives from embedding-based clusters and asks LLM to merge
    those that express the same logical reasoning, returning a refined mapping.
    """
    rids = sorted(rep_texts.keys())
    if len(rids) <= 1:
        return {rids[0]: 0} if rids else {}

    step_lines = []
    for idx, rid in enumerate(rids):
        t = rep_texts[rid][:500].replace("\n", " ")
        step_lines.append(f'[{idx}]: "{t}"')

    prompt = (
        f"I have {len(rids)} representative reasoning steps, each from a different cluster. "
        "Some clusters may actually express the SAME logical reasoning approach "
        "(same math operation, same strategy, same conclusion) despite different wording.\n\n"
        "Merge clusters that perform essentially the same reasoning into one group.\n\n"
        "Representatives:\n" + "\n".join(step_lines) + "\n\n"
        "Respond with ONLY a JSON object mapping step index (string) to final cluster label (int starting from 0). "
        "Steps that should be merged get the same label.\n"
        'Example: {"0": 0, "1": 0, "2": 1, "3": 2}'
    )

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=len(rids) * 15 + 50,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content)
                mapping = {}
                for idx_str, cluster in raw.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(rids):
                        mapping[rids[idx]] = int(cluster)
                next_c = max(mapping.values(), default=-1) + 1
                for rid in rids:
                    if rid not in mapping:
                        mapping[rid] = next_c
                        next_c += 1
                return mapping
            except Exception as e:
                if attempt == 2:
                    print(f"    LLM merge failed: {e}")
                    return {rid: i for i, rid in enumerate(rids)}
                await asyncio.sleep(2 * (attempt + 1))

    return {rid: i for i, rid in enumerate(rids)}


async def cluster_steps_semantic(
    client, step_texts, semaphore,
    embedding_model="text-embedding-3-small",
    judge_model="gpt-4o",
    embedding_threshold=0.85,
):
    """Two-phase semantic clustering:
    Phase 1: Fast embedding-based clustering (cosine similarity)
    Phase 2: LLM refinement — merge embedding clusters that express same logic
    """
    rids = sorted(step_texts.keys())
    if len(rids) <= 1:
        return {rids[0]: 0} if rids else {}

    # Fast path: exact text match
    text_groups = defaultdict(list)
    for rid in rids:
        text_groups[step_texts[rid]].append(rid)
    if len(text_groups) == 1:
        return {rid: 0 for rid in rids}

    # Phase 1: Embedding-based clustering
    texts = [step_texts[rid][:500] for rid in rids]
    async with semaphore:
        embeddings = await get_embeddings_batch(client, texts, model=embedding_model)
    labels = cluster_by_embeddings(embeddings, threshold=embedding_threshold)

    # Group by embedding cluster
    emb_clusters = defaultdict(list)
    for rid, label in zip(rids, labels):
        emb_clusters[label].append(rid)

    num_emb_clusters = len(emb_clusters)

    # If few clusters, no need for LLM refinement
    if num_emb_clusters <= 3:
        return {rid: label for rid, label in zip(rids, labels)}

    # Phase 2: LLM refinement — pick representative per cluster, ask LLM to merge
    rep_texts = {}
    cluster_members = {}  # rep_rid -> list of all rids in that cluster
    for cid, cluster_rids in emb_clusters.items():
        rep_rid = cluster_rids[0]
        rep_texts[rep_rid] = step_texts[rep_rid]
        cluster_members[rep_rid] = cluster_rids

    # If too many representatives, do multi-round merging
    rep_rids = sorted(rep_texts.keys())
    if len(rep_rids) <= 40:
        merge_mapping = await _llm_judge_merge(
            client, semaphore, rep_texts, model=judge_model
        )
    else:
        # Split representatives into groups, merge within groups, then merge across
        merge_mapping = {}
        group_size = 30
        intermediate_clusters = {}
        global_offset = 0

        for i in range(0, len(rep_rids), group_size):
            group_rids = rep_rids[i:i + group_size]
            group_texts = {rid: rep_texts[rid] for rid in group_rids}
            group_mapping = await _llm_judge_merge(
                client, semaphore, group_texts, model=judge_model
            )
            # Offset cluster IDs to avoid collisions
            for rid, cid in group_mapping.items():
                merge_mapping[rid] = cid + global_offset
                intermediate_clusters.setdefault(cid + global_offset, []).append(rid)
            if group_mapping:
                global_offset += max(group_mapping.values()) + 1

        # Second pass: merge across groups using representatives of intermediate clusters
        if len(intermediate_clusters) > 1 and len(intermediate_clusters) <= 40:
            inter_rep_texts = {rids[0]: rep_texts[rids[0]]
                                for cid, rids in intermediate_clusters.items()
                                if rids[0] in rep_texts}
            if len(inter_rep_texts) > 1:
                inter_merge = await _llm_judge_merge(
                    client, semaphore, inter_rep_texts, model=judge_model
                )
                # Remap
                remap = {}
                for inter_rep, final_cid in inter_merge.items():
                    old_cid = merge_mapping[inter_rep]
                    remap[old_cid] = final_cid
                for rid in merge_mapping:
                    old_cid = merge_mapping[rid]
                    if old_cid in remap:
                        merge_mapping[rid] = remap[old_cid]

    # Map back to all original rids
    final_mapping = {}
    cid_remap = {}
    next_final = 0

    for rep_rid in rep_rids:
        merged_cid = merge_mapping.get(rep_rid, rep_rid)
        if merged_cid not in cid_remap:
            cid_remap[merged_cid] = next_final
            next_final += 1
        final_cid = cid_remap[merged_cid]
        for rid in cluster_members[rep_rid]:
            final_mapping[rid] = final_cid

    # Fill missing
    for rid in rids:
        if rid not in final_mapping:
            final_mapping[rid] = next_final
            next_final += 1

    return final_mapping


async def cluster_steps_llm_only(client, step_texts, semaphore, model="gpt-4o-mini"):
    """Pure LLM clustering: directly ask LLM to group step texts by reasoning logic.

    No embedding pre-filtering — LLM sees all texts and decides grouping.
    For large inputs (>40 texts), splits into batches and merges across batches.
    """
    rids = sorted(step_texts.keys())
    if len(rids) <= 1:
        return {rids[0]: 0} if rids else {}

    # Exact text dedup
    text_groups = defaultdict(list)
    for rid in rids:
        text_groups[step_texts[rid]].append(rid)
    if len(text_groups) == 1:
        return {rid: 0 for rid in rids}

    # Get unique texts and their representative rids
    unique_texts = {}  # rep_rid -> text
    rid_to_rep = {}    # all rids -> their representative rid
    for text, group_rids in text_groups.items():
        rep = group_rids[0]
        unique_texts[rep] = text
        for rid in group_rids:
            rid_to_rep[rid] = rep

    unique_rids = sorted(unique_texts.keys())

    if len(unique_rids) <= 1:
        return {rid: 0 for rid in rids}

    async def _ask_llm_cluster(sub_rids, sub_texts):
        """Ask LLM to cluster a batch of step texts."""
        step_lines = []
        for idx, rid in enumerate(sub_rids):
            t = sub_texts[rid][:600].replace("\n", " ")
            step_lines.append(f'[{idx}]: "{t}"')

        prompt = (
            f"I have {len(sub_rids)} reasoning steps from different math problem solutions. "
            "Group them by their mathematical reasoning approach — steps that use the SAME "
            "mathematical operation, strategy, or logical step should be in the SAME group, "
            "even if the wording differs.\n\n"
            "Steps:\n" + "\n".join(step_lines) + "\n\n"
            "Respond with ONLY a JSON object mapping step index (string) to group label (int starting from 0). "
            "Steps with the same reasoning get the same label.\n"
            'Example: {"0": 0, "1": 0, "2": 1, "3": 2}'
        )

        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=len(sub_rids) * 15 + 100,
                        response_format={"type": "json_object"},
                    )
                    raw = json.loads(resp.choices[0].message.content)
                    mapping = {}
                    for idx_str, cluster in raw.items():
                        idx = int(idx_str)
                        if 0 <= idx < len(sub_rids):
                            mapping[sub_rids[idx]] = int(cluster)
                    # Fill missing
                    next_c = max(mapping.values(), default=-1) + 1
                    for rid in sub_rids:
                        if rid not in mapping:
                            mapping[rid] = next_c
                            next_c += 1
                    return mapping
                except Exception as e:
                    if attempt == 2:
                        print(f"    LLM cluster failed: {e}")
                        return {rid: i for i, rid in enumerate(sub_rids)}
                    await asyncio.sleep(2 * (attempt + 1))
        return {rid: i for i, rid in enumerate(sub_rids)}

    # If small enough, do in one call
    if len(unique_rids) <= 40:
        rep_mapping = await _ask_llm_cluster(unique_rids, unique_texts)
    else:
        # Split into batches, cluster within, then merge across
        batch_size = 30
        rep_mapping = {}
        intermediate_clusters = {}
        global_offset = 0

        for i in range(0, len(unique_rids), batch_size):
            batch_rids = unique_rids[i:i + batch_size]
            batch_texts = {rid: unique_texts[rid] for rid in batch_rids}
            batch_mapping = await _ask_llm_cluster(batch_rids, batch_texts)
            for rid, cid in batch_mapping.items():
                rep_mapping[rid] = cid + global_offset
                intermediate_clusters.setdefault(cid + global_offset, []).append(rid)
            if batch_mapping:
                global_offset += max(batch_mapping.values()) + 1

        # Second pass: merge across batches
        if len(intermediate_clusters) > 1 and len(intermediate_clusters) <= 40:
            inter_rep_texts = {}
            for cid, crids in intermediate_clusters.items():
                inter_rep_texts[crids[0]] = unique_texts[crids[0]]
            if len(inter_rep_texts) > 1:
                inter_rids = sorted(inter_rep_texts.keys())
                inter_merge = await _ask_llm_cluster(inter_rids, inter_rep_texts)
                remap = {}
                for inter_rid, final_cid in inter_merge.items():
                    old_cid = rep_mapping[inter_rid]
                    remap[old_cid] = final_cid
                for rid in rep_mapping:
                    old_cid = rep_mapping[rid]
                    if old_cid in remap:
                        rep_mapping[rid] = remap[old_cid]

    # Map back: unique rep -> all rids with same text
    final_mapping = {}
    cid_remap = {}
    next_final = 0
    for rep_rid in unique_rids:
        merged_cid = rep_mapping.get(rep_rid, 0)
        if merged_cid not in cid_remap:
            cid_remap[merged_cid] = next_final
            next_final += 1

    for rid in rids:
        rep = rid_to_rep[rid]
        merged_cid = rep_mapping.get(rep, 0)
        final_mapping[rid] = cid_remap[merged_cid]

    return final_mapping


async def build_tree_llm_semantic(problem_steps, client, semaphore, model="gpt-4o-mini"):
    """Build tree using pure LLM semantic clustering (no embeddings)."""
    all_rids = set(problem_steps.keys())
    max_steps = max(len(s) for s in problem_steps.values()) if problem_steps else 0

    root = {
        "rollout_ids": sorted(all_rids),
        "step_level": -1,
        "num_rollouts": len(all_rids),
        "children": [],
    }

    current_nodes = [root]
    for level in range(max_steps):
        next_nodes = []
        for node in current_nodes:
            active_rids = {
                rid for rid in node["rollout_ids"]
                if rid in problem_steps and level < len(problem_steps[rid])
            }
            # Preserve terminated rollouts as leaf children
            terminated_rids = {
                rid for rid in node["rollout_ids"]
                if rid not in active_rids
            }
            if terminated_rids:
                leaf = {
                    "rollout_ids": sorted(terminated_rids),
                    "step_level": level,
                    "cluster_id": -1,
                    "num_rollouts": len(terminated_rids),
                    "children": [],
                }
                node["children"].append(leaf)

            if not active_rids:
                continue

            step_texts = {rid: problem_steps[rid][level] for rid in active_rids}

            clustering = await cluster_steps_llm_only(
                client, step_texts, semaphore, model=model,
            )

            cluster_groups = defaultdict(set)
            for rid, cid in clustering.items():
                cluster_groups[cid].add(rid)

            for cid, cluster_rids in sorted(cluster_groups.items()):
                child = {
                    "rollout_ids": sorted(cluster_rids),
                    "step_level": level,
                    "cluster_id": cid,
                    "num_rollouts": len(cluster_rids),
                    "children": [],
                }
                node["children"].append(child)
                next_nodes.append(child)

        current_nodes = next_nodes

    return root


async def build_tree_semantic(problem_steps, client, semaphore,
                               embedding_model="text-embedding-3-small",
                               judge_model="gpt-4o"):
    """Build tree using embedding + LLM semantic clustering."""
    all_rids = set(problem_steps.keys())
    max_steps = max(len(s) for s in problem_steps.values()) if problem_steps else 0

    root = {
        "rollout_ids": sorted(all_rids),
        "step_level": -1,
        "num_rollouts": len(all_rids),
        "children": [],
    }

    current_nodes = [root]
    for level in range(max_steps):
        next_nodes = []
        for node in current_nodes:
            active_rids = {
                rid for rid in node["rollout_ids"]
                if rid in problem_steps and level < len(problem_steps[rid])
            }
            # Preserve terminated rollouts as leaf children
            terminated_rids = {
                rid for rid in node["rollout_ids"]
                if rid not in active_rids
            }
            if terminated_rids:
                leaf = {
                    "rollout_ids": sorted(terminated_rids),
                    "step_level": level,
                    "cluster_id": -1,
                    "num_rollouts": len(terminated_rids),
                    "children": [],
                }
                node["children"].append(leaf)

            if not active_rids:
                continue

            step_texts = {rid: problem_steps[rid][level] for rid in active_rids}

            clustering = await cluster_steps_semantic(
                client, step_texts, semaphore,
                embedding_model=embedding_model,
                judge_model=judge_model,
            )

            cluster_groups = defaultdict(set)
            for rid, cid in clustering.items():
                cluster_groups[cid].add(rid)

            for cid, cluster_rids in sorted(cluster_groups.items()):
                child = {
                    "rollout_ids": sorted(cluster_rids),
                    "step_level": level,
                    "cluster_id": cid,
                    "num_rollouts": len(cluster_rids),
                    "children": [],
                }
                node["children"].append(child)
                next_nodes.append(child)

        current_nodes = next_nodes

    return root


# ============================================================================
# Per-rollout correctness
# ============================================================================

def compute_rollout_correctness(rollouts, ground_truth):
    """Return dict: rollout_id -> bool (correct or not)."""
    correctness = {}
    for rollout in rollouts:
        rid = rollout["rollout_id"]
        text = rollout.get("full_text", "") or rollout.get("response", "")
        correctness[rid] = is_correct(text, ground_truth)
    return correctness


# ============================================================================
# Node accuracy computation
# ============================================================================

def collect_leaf_rollout_ids(node):
    """Collect rollout_ids from all leaf descendants of a node."""
    if not node["children"]:
        return set(node["rollout_ids"])
    result = set()
    for child in node["children"]:
        result |= collect_leaf_rollout_ids(child)
    return result


def compute_node_accuracies(tree, correctness):
    """For each node in the tree, compute accuracy from its leaf descendants.

    Returns list of dicts with: depth, num_rollouts, num_correct, accuracy, is_leaf
    """
    node_stats = []

    def traverse(node, depth):
        rids = collect_leaf_rollout_ids(node)
        if not rids:
            return

        num_correct = sum(1 for rid in rids if correctness.get(rid, False))
        acc = num_correct / len(rids) if rids else 0.0

        node_stats.append({
            "depth": depth,
            "num_rollouts": len(rids),
            "num_correct": num_correct,
            "accuracy": acc,
            "is_leaf": len(node["children"]) == 0,
            "cluster_id": node.get("cluster_id", -1),
        })

        for child in node["children"]:
            traverse(child, depth + 1)

    traverse(tree, 0)
    return node_stats


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(syntactic_results, semantic_results, output_dir, thresholds,
                    comparison_threshold=None):
    """Generate all comparison plots between syntactic and semantic clustering."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all node accuracy stats across problems
    def gather_stats(results_dict):
        """Flatten node stats across all problems."""
        all_stats = []
        for prob_id, stats in results_dict.items():
            for s in stats:
                s["problem_id"] = prob_id
                all_stats.append(s)
        return all_stats

    # --- PLOT 1: Scatter — node accuracy vs depth (syntactic vs semantic side by side) ---
    best_threshold = comparison_threshold if comparison_threshold is not None else thresholds[0]
    syn_stats = gather_stats(syntactic_results[best_threshold])
    sem_stats = gather_stats(semantic_results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, stats, title, color in [
        (axes[0], syn_stats, f"Syntactic (threshold={best_threshold})", "#4C72B0"),
        (axes[1], sem_stats, "Semantic (LLM)", "#C44E52"),
    ]:
        depths = [s["depth"] for s in stats if not s["is_leaf"] or s["depth"] > 0]
        accs = [s["accuracy"] for s in stats if not s["is_leaf"] or s["depth"] > 0]
        sizes = [s["num_rollouts"] for s in stats if not s["is_leaf"] or s["depth"] > 0]

        # Normalize sizes for scatter
        max_size = max(sizes) if sizes else 1
        scatter_sizes = [20 + 200 * (s / max_size) for s in sizes]

        ax.scatter(depths, accs, s=scatter_sizes, alpha=0.3, c=color, edgecolors="none")
        ax.set_xlabel("Tree Depth", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, max(depths) + 0.5 if depths else 5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.2)

        # Add mean accuracy line per depth
        depth_acc = defaultdict(list)
        for d, a in zip(depths, accs):
            depth_acc[d].append(a)
        mean_depths = sorted(depth_acc.keys())
        mean_accs = [np.mean(depth_acc[d]) for d in mean_depths]
        ax.plot(mean_depths, mean_accs, "o-", color="black", linewidth=2, markersize=5,
                label="Mean accuracy", zorder=5)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Node Accuracy\n(fraction of correct leaf descendants)", fontsize=11)
    fig.suptitle("Node Accuracy vs Tree Depth: Syntactic vs Semantic Clustering",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_accuracy_vs_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- PLOT 2: Histogram — distribution of node accuracies (overlay) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2a: All nodes
    bins = np.linspace(0, 1, 21)
    syn_accs = [s["accuracy"] for s in syn_stats if s["depth"] > 0]
    sem_accs = [s["accuracy"] for s in sem_stats if s["depth"] > 0]
    axes[0].hist(syn_accs, bins=bins, alpha=0.6, color="#4C72B0", label=f"Syntactic (t={best_threshold})", edgecolor="black")
    axes[0].hist(sem_accs, bins=bins, alpha=0.6, color="#C44E52", label="Semantic (LLM)", edgecolor="black")
    axes[0].set_xlabel("Node Accuracy", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("All Internal Nodes", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # 2b: Early depths (0-3)
    syn_early = [s["accuracy"] for s in syn_stats if 0 < s["depth"] <= 3]
    sem_early = [s["accuracy"] for s in sem_stats if 0 < s["depth"] <= 3]
    axes[1].hist(syn_early, bins=bins, alpha=0.6, color="#4C72B0", label=f"Syntactic (t={best_threshold})", edgecolor="black")
    axes[1].hist(sem_early, bins=bins, alpha=0.6, color="#C44E52", label="Semantic (LLM)", edgecolor="black")
    axes[1].set_xlabel("Node Accuracy", fontsize=11)
    axes[1].set_title("Early Depths (1-3)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    # 2c: Late depths (4+)
    syn_late = [s["accuracy"] for s in syn_stats if s["depth"] > 3]
    sem_late = [s["accuracy"] for s in sem_stats if s["depth"] > 3]
    axes[2].hist(syn_late, bins=bins, alpha=0.6, color="#4C72B0", label=f"Syntactic (t={best_threshold})", edgecolor="black")
    axes[2].hist(sem_late, bins=bins, alpha=0.6, color="#C44E52", label="Semantic (LLM)", edgecolor="black")
    axes[2].set_xlabel("Node Accuracy", fontsize=11)
    axes[2].set_title("Late Depths (4+)", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.2)

    fig.suptitle("Distribution of Node Accuracies: Syntactic vs Semantic", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "histogram_accuracy_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- PLOT 3: Box plot — accuracy by depth for both methods ---
    max_depth_plot = min(12, max(
        max((s["depth"] for s in syn_stats), default=0),
        max((s["depth"] for s in sem_stats), default=0),
    ))

    fig, ax = plt.subplots(figsize=(14, 6))
    positions_syn = []
    positions_sem = []
    data_syn = []
    data_sem = []
    labels = []

    for d in range(1, max_depth_plot + 1):
        syn_d = [s["accuracy"] for s in syn_stats if s["depth"] == d]
        sem_d = [s["accuracy"] for s in sem_stats if s["depth"] == d]
        if syn_d or sem_d:
            data_syn.append(syn_d if syn_d else [0])
            data_sem.append(sem_d if sem_d else [0])
            positions_syn.append(d * 3 - 0.4)
            positions_sem.append(d * 3 + 0.4)
            labels.append(f"Depth {d}")

    bp1 = ax.boxplot(data_syn, positions=positions_syn, widths=0.6,
                      patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_sem, positions=positions_sem, widths=0.6,
                      patch_artist=True, showfliers=False)

    for patch in bp1["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#C44E52")
        patch.set_alpha(0.7)

    ax.set_xticks([d * 3 for d in range(1, max_depth_plot + 1) if d <= len(labels)])
    ax.set_xticklabels(labels[:max_depth_plot], rotation=45, ha="right")
    ax.set_ylabel("Node Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.2, axis="y")

    syn_patch = mpatches.Patch(color="#4C72B0", alpha=0.7, label=f"Syntactic (t={best_threshold})")
    sem_patch = mpatches.Patch(color="#C44E52", alpha=0.7, label="Semantic (LLM)")
    ax.legend(handles=[syn_patch, sem_patch], fontsize=10)
    ax.set_title("Node Accuracy by Depth: Syntactic vs Semantic Clustering", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_accuracy_by_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- PLOT 4: Syntactic threshold sweep ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 4a: Number of clusters at depth 1 vs threshold
    threshold_clusters = {}
    for t in thresholds:
        all_depth1_clusters = []
        for prob_id, stats in syntactic_results[t].items():
            depth1_nodes = [s for s in stats if s["depth"] == 1]
            all_depth1_clusters.append(len(depth1_nodes))
        threshold_clusters[t] = np.mean(all_depth1_clusters) if all_depth1_clusters else 0

    axes[0].plot(thresholds, [threshold_clusters[t] for t in thresholds],
                 "o-", color="#4C72B0", linewidth=2, markersize=8)
    axes[0].set_xlabel("Similarity Threshold", fontsize=11)
    axes[0].set_ylabel("Avg Clusters at Depth 1", fontsize=11)
    axes[0].set_title("Syntactic: Threshold vs Granularity", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 4b: Mean node accuracy variance vs threshold (higher variance = more discriminative)
    threshold_acc_var = {}
    for t in thresholds:
        all_accs = [s["accuracy"] for s in gather_stats(syntactic_results[t]) if s["depth"] > 0]
        threshold_acc_var[t] = np.var(all_accs) if all_accs else 0

    axes[1].plot(thresholds, [threshold_acc_var[t] for t in thresholds],
                 "s-", color="#55A868", linewidth=2, markersize=8)
    axes[1].set_xlabel("Similarity Threshold", fontsize=11)
    axes[1].set_ylabel("Variance of Node Accuracy", fontsize=11)
    axes[1].set_title("Syntactic: Threshold vs Accuracy Discrimination", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Effect of Syntactic Similarity Threshold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- PLOT 5: Per-problem scatter — syntactic vs semantic accuracy correlation ---
    # For each problem, compute mean node accuracy for both methods
    prob_ids_common = sorted(set(syntactic_results[best_threshold].keys()) &
                              set(semantic_results.keys()))

    fig, ax = plt.subplots(figsize=(8, 8))
    syn_prob_accs = []
    sem_prob_accs = []
    prob_levels = []

    for pid in prob_ids_common:
        syn_nodes = [s["accuracy"] for s in syntactic_results[best_threshold][pid] if s["depth"] > 0]
        sem_nodes = [s["accuracy"] for s in semantic_results[pid] if s["depth"] > 0]
        syn_mean = np.mean(syn_nodes) if syn_nodes else 0
        sem_mean = np.mean(sem_nodes) if sem_nodes else 0
        syn_prob_accs.append(syn_mean)
        sem_prob_accs.append(sem_mean)

    ax.scatter(syn_prob_accs, sem_prob_accs, s=100, c="#8856a7", alpha=0.7, edgecolors="black")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x")

    for i, pid in enumerate(prob_ids_common):
        ax.annotate(f"P{pid}", (syn_prob_accs[i], sem_prob_accs[i]),
                    fontsize=7, ha="left", va="bottom")

    corr = np.corrcoef(syn_prob_accs, sem_prob_accs)[0, 1] if len(prob_ids_common) > 1 else 0
    ax.set_xlabel(f"Mean Node Accuracy (Syntactic, t={best_threshold})", fontsize=11)
    ax.set_ylabel("Mean Node Accuracy (Semantic, LLM)", fontsize=11)
    ax.set_title(f"Per-Problem Accuracy Correlation\nPearson r = {corr:.3f}", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_per_problem_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- PLOT 6: Accuracy spread at early branching (depth 1) ---
    # This is the key plot: at the first branching point, do different branches
    # lead to different accuracy outcomes? If yes, tree structure captures signal.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, results_dict, title, color in [
        (axes[0], syntactic_results[best_threshold], f"Syntactic (t={best_threshold})", "#4C72B0"),
        (axes[1], semantic_results, "Semantic (LLM)", "#C44E52"),
    ]:
        prob_data = []
        prob_labels = []
        for pid in sorted(results_dict.keys()):
            depth1_accs = [s["accuracy"] for s in results_dict[pid] if s["depth"] == 1]
            if len(depth1_accs) >= 2:
                prob_data.append(depth1_accs)
                prob_labels.append(f"P{pid}")

        if prob_data:
            bp = ax.boxplot(prob_data, patch_artist=True, showfliers=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticklabels(prob_labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Problem", fontsize=11)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
            ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Branch Accuracy at Depth 1", fontsize=11)
    fig.suptitle("Accuracy Spread at First Branching Point\n"
                 "(Do different branches lead to different outcomes?)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_depth1_branching.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved 6 plots to {output_dir}/")
    return {
        "correlation": float(corr),
        "syn_mean_accuracy": float(np.mean(syn_prob_accs)) if syn_prob_accs else 0,
        "sem_mean_accuracy": float(np.mean(sem_prob_accs)) if sem_prob_accs else 0,
    }


# ============================================================================
# Main pipeline
# ============================================================================

async def run_analysis(args):
    """Main analysis pipeline."""
    # Load flat rollout results
    print(f"Loading flat rollout results from {args.flat_results}...")
    with open(args.flat_results) as f:
        flat_data = json.load(f)

    results = flat_data["results"]
    config = flat_data.get("config", {})
    model_name = config.get("model", "unknown")

    if args.num_problems:
        results = results[:args.num_problems]
    print(f"  {len(results)} problems loaded")

    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                                   trust_remote_code=True)

    # Chunk all rollouts into steps and compute correctness
    print(f"Chunking rollouts into {args.step_size}-token steps...")
    all_problem_steps = []
    all_correctness = []
    for entry in results:
        problem_steps = {}
        for rollout in entry["rollouts"]:
            rid = rollout["rollout_id"]
            text = rollout.get("thinking", "") or rollout.get("full_text", "") or rollout.get("response", "")
            steps = chunk_text_into_steps(text, tokenizer, step_size=args.step_size)
            if steps:
                problem_steps[rid] = steps
        all_problem_steps.append(problem_steps)
        all_correctness.append(
            compute_rollout_correctness(entry["rollouts"], entry.get("answer", ""))
        )

    # ==========================================
    # STEP 1: Build trees with both methods
    # ==========================================

    # 1a: Syntactic matching with multiple thresholds
    thresholds = [float(t) for t in args.syntactic_thresholds.split(",")]
    print(f"\n{'='*60}")
    print("STEP 1a: Syntactic (substring) matching")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*60}")

    syntactic_trees = {}  # threshold -> {prob_id -> tree}
    syntactic_node_stats = {}  # threshold -> {prob_id -> [node_stats]}

    for threshold in thresholds:
        print(f"\n  Threshold = {threshold}")
        trees_for_threshold = {}
        stats_for_threshold = {}
        t0 = time.time()

        for i, (problem_steps, correctness) in enumerate(zip(all_problem_steps, all_correctness)):
            tree = await build_tree_for_problem(
                problem_steps, None, None,
                use_llm=False,
                similarity_threshold=threshold,
            )
            trees_for_threshold[i] = tree
            stats_for_threshold[i] = compute_node_accuracies(tree, correctness)

            # Count nodes at depth 1 for progress
            n_depth1 = len([s for s in stats_for_threshold[i] if s["depth"] == 1])
            print(f"    Problem {i}: {n_depth1} branches at depth 1, "
                  f"{len(stats_for_threshold[i])} total nodes")

        syntactic_trees[threshold] = trees_for_threshold
        syntactic_node_stats[threshold] = stats_for_threshold
        print(f"  Threshold {threshold} done in {time.time() - t0:.1f}s")

    # 1b: Semantic matching with LLM
    print(f"\n{'='*60}")
    print("STEP 1b: Semantic (LLM) matching")
    print(f"{'='*60}")

    semantic_trees = {}
    semantic_node_stats = {}

    if args.use_llm:
        from openai import AsyncOpenAI
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  ERROR: No OpenAI API key. Set --openai_api_key or OPENAI_API_KEY env var.")
            print("  Falling back to proxy mode.")
            args.use_llm = False

    if args.use_llm:
        client = AsyncOpenAI(api_key=api_key)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        t0 = time.time()

        print(f"  Embedding model: {args.embedding_model}")
        print(f"  Judge model: {args.judge_model}")
        print(f"  Embedding threshold: {args.embedding_threshold}")

        for i, (problem_steps, correctness) in enumerate(zip(all_problem_steps, all_correctness)):
            tree = await build_tree_semantic(
                problem_steps, client, semaphore,
                embedding_model=args.embedding_model,
                judge_model=args.judge_model,
            )
            semantic_trees[i] = tree
            semantic_node_stats[i] = compute_node_accuracies(tree, correctness)

            n_depth1 = len([s for s in semantic_node_stats[i] if s["depth"] == 1])
            print(f"  Problem {i}: {n_depth1} branches at depth 1, "
                  f"{len(semantic_node_stats[i])} total nodes")

        print(f"  Semantic matching done in {time.time() - t0:.1f}s")
    else:
        print("  LLM not available — using highest syntactic threshold as proxy")
        highest_t = max(thresholds)
        semantic_node_stats = syntactic_node_stats[highest_t]
        semantic_trees = syntactic_trees[highest_t]

    # ==========================================
    # STEP 2: Already computed in compute_node_accuracies above
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 2: Node accuracy statistics")
    print(f"{'='*60}")

    # Print summary for each method
    for t in thresholds:
        all_accs = [s["accuracy"] for pid, stats in syntactic_node_stats[t].items()
                     for s in stats if s["depth"] > 0]
        print(f"  Syntactic t={t}: {len(all_accs)} nodes, "
              f"mean_acc={np.mean(all_accs):.3f}, var={np.var(all_accs):.4f}")

    sem_all_accs = [s["accuracy"] for pid, stats in semantic_node_stats.items()
                     for s in stats if s["depth"] > 0]
    method_name = "Semantic (LLM)" if args.use_llm else f"Semantic (proxy t={max(thresholds)})"
    print(f"  {method_name}: {len(sem_all_accs)} nodes, "
          f"mean_acc={np.mean(sem_all_accs):.3f}, var={np.var(sem_all_accs):.4f}")

    # ==========================================
    # STEP 3: Visualization
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 3: Generating visualizations")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    comparison_t = args.comparison_threshold if args.comparison_threshold else thresholds[0]
    plot_stats = plot_comparison(
        syntactic_node_stats, semantic_node_stats,
        args.output_dir, thresholds,
        comparison_threshold=comparison_t,
    )

    # Save detailed results JSON
    summary = {
        "config": {
            "model": model_name,
            "num_problems": len(results),
            "step_size": args.step_size,
            "syntactic_thresholds": thresholds,
            "semantic_method": args.eval_model if args.use_llm else f"proxy_t={max(thresholds)}",
        },
        "syntactic": {},
        "semantic": {},
        "comparison": plot_stats,
    }

    for t in thresholds:
        all_accs = [s["accuracy"] for pid, stats in syntactic_node_stats[t].items()
                     for s in stats if s["depth"] > 0]
        depth1_accs = [s["accuracy"] for pid, stats in syntactic_node_stats[t].items()
                        for s in stats if s["depth"] == 1]
        summary["syntactic"][str(t)] = {
            "total_nodes": len(all_accs),
            "mean_accuracy": float(np.mean(all_accs)) if all_accs else 0,
            "accuracy_variance": float(np.var(all_accs)) if all_accs else 0,
            "depth1_nodes": len(depth1_accs),
            "depth1_mean_acc": float(np.mean(depth1_accs)) if depth1_accs else 0,
            "depth1_acc_variance": float(np.var(depth1_accs)) if depth1_accs else 0,
            "per_problem": {
                str(pid): {
                    "num_nodes": len(stats),
                    "depth1_branches": len([s for s in stats if s["depth"] == 1]),
                    "mean_accuracy": float(np.mean([s["accuracy"] for s in stats if s["depth"] > 0]))
                    if any(s["depth"] > 0 for s in stats) else 0,
                }
                for pid, stats in syntactic_node_stats[t].items()
            },
        }

    sem_depth1 = [s["accuracy"] for pid, stats in semantic_node_stats.items()
                   for s in stats if s["depth"] == 1]
    summary["semantic"] = {
        "total_nodes": len(sem_all_accs),
        "mean_accuracy": float(np.mean(sem_all_accs)) if sem_all_accs else 0,
        "accuracy_variance": float(np.var(sem_all_accs)) if sem_all_accs else 0,
        "depth1_nodes": len(sem_depth1),
        "depth1_mean_acc": float(np.mean(sem_depth1)) if sem_depth1 else 0,
        "depth1_acc_variance": float(np.var(sem_depth1)) if sem_depth1 else 0,
        "per_problem": {
            str(pid): {
                "num_nodes": len(stats),
                "depth1_branches": len([s for s in stats if s["depth"] == 1]),
                "mean_accuracy": float(np.mean([s["accuracy"] for s in stats if s["depth"] > 0]))
                if any(s["depth"] > 0 for s in stats) else 0,
            }
            for pid, stats in semantic_node_stats.items()
        },
    }

    summary_path = os.path.join(args.output_dir, "node_accuracy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Correlation (syntactic vs semantic): {plot_stats['correlation']:.3f}")
    print(f"Syntactic mean node accuracy: {plot_stats['syn_mean_accuracy']:.3f}")
    print(f"Semantic mean node accuracy:  {plot_stats['sem_mean_accuracy']:.3f}")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - scatter_accuracy_vs_depth.png")
    print(f"  - histogram_accuracy_distribution.png")
    print(f"  - boxplot_accuracy_by_depth.png")
    print(f"  - threshold_sweep.png")
    print(f"  - scatter_per_problem_correlation.png")
    print(f"  - boxplot_depth1_branching.png")
    print(f"  - node_accuracy_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Node-level accuracy analysis: syntactic vs semantic tree clustering")
    parser.add_argument("--flat_results", required=True,
                        help="Path to flat rollout results JSON")
    parser.add_argument("--output_dir", default="faithful_baseline/results/node_accuracy",
                        help="Output directory")
    parser.add_argument("--step_size", type=int, default=256,
                        help="Tokens per step (default: 256)")
    parser.add_argument("--num_problems", type=int, default=None,
                        help="Limit number of problems")
    parser.add_argument("--syntactic_thresholds", type=str, default="0.3,0.5,0.6,0.7,0.9",
                        help="Comma-separated syntactic similarity thresholds")
    parser.add_argument("--use_llm", action="store_true", default=False,
                        help="Use embedding+LLM for semantic clustering")
    parser.add_argument("--embedding_model", default="text-embedding-3-small",
                        help="OpenAI embedding model for phase 1")
    parser.add_argument("--embedding_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for embedding clustering")
    parser.add_argument("--judge_model", default="gpt-4o",
                        help="OpenAI model for LLM judge refinement (phase 2)")
    parser.add_argument("--eval_model", default="gpt-4o",
                        help="Alias for judge_model (backward compat)")
    parser.add_argument("--openai_api_key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max_concurrent", type=int, default=30,
                        help="Max concurrent API calls")
    parser.add_argument("--comparison_threshold", type=float, default=None,
                        help="Syntactic threshold to use in comparison plots (default: first threshold)")
    args = parser.parse_args()

    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
