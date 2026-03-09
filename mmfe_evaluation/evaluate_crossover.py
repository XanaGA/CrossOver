#!/usr/bin/env python3
"""
Evaluate CrossOver scene retrieval on Structured3D.

Given a query modality and a database modality, encodes all scenes with
CrossOver's floorplan encoder (all S3D modalities are 2D images) and
computes cross-modal retrieval accuracy using cosine similarity.

Usage
-----
python mmfe_evaluation/evaluate_crossover.py \
    --ckpt /path/to/crossover_checkpoint \
    --data_root /path/to/Structured3D \
    --val_json /path/to/val.json \
    --query_modality floorplan \
    --database_modality lidar \
    --num_examples 100 \
    --batch_size 16

Add ``--all_pairs`` to evaluate every ordered modality pair.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.scene_crossover import SceneCrossOverModel
from util import torch_util
from mmfe_evaluation.dual_transforms import (
    PairToPIL,
    PairResize,
    PairToTensor,
    PairNormalize,
)
from mmfe_evaluation.s3d_data import Structured3DDataset
from mmfe_evaluation.unified_dataset import UnifiedDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: Tuple[int, int]):
    """CrossOver-compatible image preprocessing (same as demo floorplan path)."""
    return [
        PairToPIL(),
        PairResize(image_size),
        PairToTensor(),
        PairNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def skip_none_collate(batch):
    """Filter out ``None`` items (scenes with missing modalities) then collate."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


def compute_all_embeddings(
    model: SceneCrossOverModel,
    dataloader: DataLoader,
    num_examples: Optional[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode both modalities for every example using ``model.encode_floorplan``.

    If *num_examples* is ``None`` the entire dataset is processed.
    Returns two tensors of shape ``(N, D)`` — one per modality.
    """
    emb0_parts: List[torch.Tensor] = []
    emb1_parts: List[torch.Tensor] = []

    batch_size = dataloader.batch_size or 1

    if num_examples is not None:
        num_batches = (num_examples + batch_size - 1) // batch_size
        total_desc = (
            f"up to {num_batches * batch_size} examples "
            f"(requested {num_examples})"
        )
    else:
        num_batches = None
        total_desc = f"all {len(dataloader.dataset)} examples"

    print(f"Computing embeddings for {total_desc} ...")

    skipped = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if num_batches is not None and i >= num_batches:
                break
            if batch is None:
                skipped += 1
                continue

            img0 = batch["modality_0"].to(device)
            img1 = batch["modality_1"].to(device)

            e0 = model.encode_floorplan(img0)
            e1 = model.encode_floorplan(img1)

            emb0_parts.append(e0.cpu())
            emb1_parts.append(e1.cpu())

    if skipped:
        print(f"  (skipped {skipped} batches due to missing modalities)")

    if not emb0_parts:
        raise RuntimeError(
            "No valid samples found — all scenes had missing modalities."
        )

    embeddings_0 = torch.cat(emb0_parts, dim=0)
    embeddings_1 = torch.cat(emb1_parts, dim=0)
    print(
        f"Embeddings: query {embeddings_0.shape}, database {embeddings_1.shape}"
    )
    return embeddings_0, embeddings_1


def retrieve(
    queries: torch.Tensor, database: torch.Tensor, max_topk: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Cosine-similarity retrieval. Each query i has ground-truth match i.
    """
    q_norm = F.normalize(queries, dim=1)
    d_norm = F.normalize(database, dim=1)
    sim = q_norm @ d_norm.T

    topk = min(max_topk, sim.shape[1])
    _, topk_idx = torch.topk(sim, k=topk, dim=1)

    return {
        "predicted_idx": sim.argmax(dim=1),
        "correct_idx": torch.arange(queries.shape[0]),
        "topk_indices": topk_idx,
    }


def compute_metrics(
    results: Dict[str, Dict[str, torch.Tensor]],
    topk_list: List[int],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for direction, res in results.items():
        pred = res["predicted_idx"].numpy()
        correct = res["correct_idx"].numpy()
        topk_idx = res["topk_indices"].numpy()
        N = len(pred)

        m: Dict[str, float] = {"num_examples": N}
        for k in topk_list:
            k_slice = topk_idx[:, : min(k, topk_idx.shape[1])]
            hits = np.any(k_slice == correct[:, None], axis=1)
            m[f"top{k}"] = 100.0 * hits.sum() / N
        metrics[direction] = m
    return metrics


def print_metrics(
    metrics: Dict[str, Dict[str, float]],
    topk_list: List[int],
) -> None:
    print("\n" + "=" * 70)
    print("CROSSOVER RETRIEVAL EVALUATION")
    print("=" * 70)
    for direction, m in metrics.items():
        print(f"\n  {direction}  (N={int(m['num_examples'])})")
        for k in topk_list:
            key = f"top{k}"
            print(f"    Top-{k}: {m.get(key, 0):.2f}%")
    print("\n" + "=" * 70)


def evaluate_pair(
    model: SceneCrossOverModel,
    args: argparse.Namespace,
    query_mod: str,
    db_mod: str,
    device: str,
    topk_list: List[int],
) -> Dict[str, Dict[str, float]]:
    """Run a full evaluation for one (query, database) modality pair."""

    transforms = build_transforms(tuple(args.image_size))

    s3d = Structured3DDataset(
        root_dir=args.data_root,
        scene_ids_file=args.val_json,
        dual_transform=transforms,
        modality_pairs=[(query_mod, db_mod)],
        furniture_pct=args.furniture_pct,
    )

    dataset = UnifiedDataset(datasets=[s3d])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=skip_none_collate,
    )

    num_examples = args.num_examples if args.num_examples is not None else len(dataset)
    print(f"\nDataset: {len(dataset)} scenes  |  {query_mod} -> {db_mod}")

    emb_q, emb_db = compute_all_embeddings(model, loader, num_examples, device)

    max_topk = max(topk_list)
    results = {
        f"{query_mod} -> {db_mod}": retrieve(emb_q, emb_db, max_topk),
        f"{db_mod} -> {query_mod}": retrieve(emb_db, emb_q, max_topk),
    }
    return compute_metrics(results, topk_list)


def evaluate_mmfe(
    model: SceneCrossOverModel,
    args: argparse.Namespace,
    device: str,
    topk_list: List[int],
) -> Dict[str, Dict[str, float]]:
    """MMFE-style evaluation: unified dataset with all modality pairs mixed."""

    transforms = build_transforms(tuple(args.image_size))

    s3d = Structured3DDataset(
        root_dir=args.data_root,
        scene_ids_file=args.val_json,
        dual_transform=transforms,
        modality_pairs=None,  # all non-identical pairs
        furniture_pct=args.furniture_pct,
    )

    dataset = UnifiedDataset(datasets=[s3d])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=skip_none_collate,
    )

    num_examples = args.num_examples if args.num_examples is not None else len(dataset)
    print(f"\nMMFE evaluation: {len(dataset)} scenes  |  all modality pairs")

    emb_q, emb_db = compute_all_embeddings(model, loader, num_examples, device)

    max_topk = max(topk_list)
    results = {
        "unified (mixed modalities)": retrieve(emb_q, emb_db, max_topk),
    }
    return compute_metrics(results, topk_list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrossOver retrieval on Structured3D")

    p.add_argument("--ckpt", type=str, required=True,
                    help="Directory containing model.safetensors")
    p.add_argument("--data_root", type=str, required=True,
                    help="Structured3D dataset root")
    p.add_argument("--val_json", type=str, default=None,
                    help="Validation split file (.json or .txt)")

    p.add_argument("--query_modality", type=str, default="floorplan",
                    choices=["floorplan", "lidar", "density_map"])
    p.add_argument("--database_modality", type=str, default="lidar",
                    choices=["floorplan", "lidar", "density_map"])
    p.add_argument("--all_pairs", action="store_true",
                    help="Evaluate all ordered modality pairs")
    p.add_argument("--mmfe_evaluation", action="store_true",
                    help="MMFE-style evaluation: unified dataset with all "
                         "modality pairs (no fixed query/database modality)")

    p.add_argument("--num_examples", type=int, default=None,
                    help="Number of examples to evaluate (default: entire dataset)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--furniture_pct", type=float, nargs="+", default=[0.0])
    p.add_argument("--output_dir", type=str, default="outputs/crossover_eval")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--input_dim_3d", type=int, default=512)
    p.add_argument("--input_dim_2d", type=int, default=1536)
    p.add_argument("--input_dim_1d", type=int, default=768)
    p.add_argument("--out_dim", type=int, default=768)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading CrossOver model from {args.ckpt} ...")
    model = SceneCrossOverModel(args, device)
    model.to(device)
    torch_util.load_weights(model, args.ckpt, device)
    model.eval()
    print("Model ready.\n")

    topk_list = [1, 5, 10]
    all_metrics: Dict[str, Dict[str, float]] = {}

    if args.mmfe_evaluation and args.all_pairs:
        raise SystemExit(
            "Error: --mmfe_evaluation and --all_pairs are mutually exclusive."
        )

    modalities = ["floorplan", "lidar", "density_map"]

    if args.mmfe_evaluation:
        pair_metrics = evaluate_mmfe(model, args, device, topk_list)
        all_metrics.update(pair_metrics)
    elif args.all_pairs:
        pairs = [(a, b) for a in modalities for b in modalities if a != b]
        for q_mod, db_mod in pairs:
            pair_metrics = evaluate_pair(
                model, args, q_mod, db_mod, device, topk_list
            )
            all_metrics.update(pair_metrics)
    else:
        pair_metrics = evaluate_pair(
            model, args, args.query_modality, args.database_modality,
            device, topk_list,
        )
        all_metrics.update(pair_metrics)

    print_metrics(all_metrics, topk_list)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
