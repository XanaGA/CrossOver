#!/usr/bin/env python3
"""
Multi-checkpoint retrieval evaluation script

This script:
1) Finds the best checkpoint for each experiment ID (lowest val_loss, highest epoch on tie)
2) Runs retrieval evaluation for each checkpoint across configured modes
3) Saves results with clear naming for comparison and prints a compact summary table

Usage:
    python scripts/evaluation_scripts/run_multiple_retrieval.py
"""

import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Experiment IDs to evaluate (only used when evaluating contrastive checkpoints)
EXPERIMENT_IDS = ["47853216"]

# Base paths
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_RETRIEVAL_SCRIPT = REPO_ROOT / "scripts/evaluation_scripts/evaluate_retrieval.py"
BASE_CHECKPOINT_DIR = str(REPO_ROOT / "outputs/contrastive/checkpoints/remote")
BASE_OUTPUT_DIR = str(REPO_ROOT / "outputs/metrics/retrieval_runs")

# Datasets
DATASETS = ["cubicasa", "structured3d", "scannet"]
DATASETS_DIRS = {
    "cubicasa": str(REPO_ROOT / "data/cubicasa5k"),
    "structured3d": str(REPO_ROOT / "data/structure3D/Structured3D_bbox/Structured3D"),
    "scannet": str(REPO_ROOT / "data/scannet/rendered"),
    "aria_synthenv": str(REPO_ROOT / "data/aria/SyntheticEnv/rendered_data"),
    "swiss_dwellings": str(REPO_ROOT / "data/SwissDwellings/modified-swiss-dwellings-v2/all_render"),
    "zillow": str(REPO_ROOT / "data/zind/rendered_data"),
}
DATASETS_FILES = {
    "cubicasa": str(REPO_ROOT / "data/cubicasa5k/val.txt"),
    "structured3d": str(REPO_ROOT / "data/structure3D/val.json"),
    "scannet": str(REPO_ROOT / "data/scannet/val.txt"),
    "aria_synthenv": str(REPO_ROOT / "data/aria/SyntheticEnv/val.txt"),
    "swiss_dwellings": str(REPO_ROOT / "data/SwissDwellings/modified-swiss-dwellings-v2/val.txt"),
    "zillow": str(REPO_ROOT / "data/zind/val.txt"),
}

# Retrieval configs to sweep
RETRIEVAL_MODES = ["1d_vectors", "voting"]  # Supported by evaluate_retrieval.py
NUM_EXAMPLES = [4000]
NUM_VOTES = [50]  # Only used for voting mode
NON_VOTING_PLACEHOLDER_VOTES = 0  # evaluate_retrieval expects this field even for 1d mode
TOPK_METRICS = [1, 5, 10]

# Device
DEVICE = "cuda"

# Optional: evaluate the SALAD descriptor directly (no checkpoint search)
EVALUATE_SALAD = False
SALAD_MODEL_CFG = {
    "checkpoint": "salad",
    # These two are read by the loader via cfg.model
    "local_path": "/home/xavi/salad",
    "salad_weights_path": "/home/xavi/mmfe/data/checkpoints/salad_checkpoints/dino_salad.ckpt",
}


def parse_bool(value: str) -> bool:
    """Parse a string into a boolean."""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")


def parse_key_value_list(pairs: List[str]) -> Dict[str, str]:
    """Parse KEY=VALUE CLI arguments into a dictionary."""
    result: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE format, got '{pair}'")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Empty key in pair '{pair}'")
        result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for configuring the retrieval sweep."""
    parser = argparse.ArgumentParser(description="Run multi-checkpoint retrieval evaluation with configurable options.")

    default_dataset_dirs = [f"{k}={v}" for k, v in DATASETS_DIRS.items()]
    default_dataset_files = [f"{k}={v}" for k, v in DATASETS_FILES.items()]

    parser.add_argument("--experiment-ids", nargs="+", default=EXPERIMENT_IDS, help="Experiment IDs to evaluate.")
    parser.add_argument("--base-checkpoint-dir", type=str, default=BASE_CHECKPOINT_DIR, help="Directory containing experiment checkpoint folders.")
    parser.add_argument("--base-output-dir", type=str, default=BASE_OUTPUT_DIR, help="Directory to save retrieval results.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to evaluate.")
    parser.add_argument("--datasets-dirs", nargs="+", default=default_dataset_dirs, metavar="NAME=PATH", help="Dataset directory overrides (NAME=PATH).")
    parser.add_argument("--datasets-files", nargs="+", default=default_dataset_files, metavar="NAME=VAL_FILE", help="Dataset val file overrides (NAME=FILE).")
    parser.add_argument("--retrieval-modes", nargs="+", default=RETRIEVAL_MODES, help="Retrieval modes to evaluate.")
    parser.add_argument("--num-examples", nargs="+", type=int, default=NUM_EXAMPLES, help="Number of examples per run.")
    parser.add_argument("--num-votes", nargs="+", type=int, default=NUM_VOTES, help="Vote counts for voting mode.")
    parser.add_argument("--topk-metrics", nargs="+", type=int, default=TOPK_METRICS, help="Top-k accuracies to compute (e.g., 1 5 10).")
    parser.add_argument("--device", type=str, default=DEVICE, help="Computation device passed to Hydra (runtime.device).")
    parser.add_argument("--evaluate-salad", type=parse_bool, default=EVALUATE_SALAD, help="Whether to include the SALAD descriptor baseline.")
    parser.add_argument("--salad-checkpoint", type=str, default=SALAD_MODEL_CFG.get("checkpoint"), help="SALAD checkpoint identifier.")
    parser.add_argument("--salad-local-path", type=str, default=SALAD_MODEL_CFG.get("local_path"), help="Local path for SALAD repo.")
    parser.add_argument("--salad-weights-path", type=str, default=SALAD_MODEL_CFG.get("salad_weights_path"), help="Path to SALAD weights checkpoint.")

    args = parser.parse_args()

    args.datasets_dirs = parse_key_value_list(args.datasets_dirs)
    args.datasets_files = parse_key_value_list(args.datasets_files)

    missing_dirs = [d for d in args.datasets if d not in args.datasets_dirs]
    missing_files = [d for d in args.datasets if d not in args.datasets_files]
    if missing_dirs:
        parser.error(f"Missing dataset directories for: {missing_dirs}. Provide via --datasets-dirs NAME=PATH.")
    if missing_files:
        parser.error(f"Missing dataset val files for: {missing_files}. Provide via --datasets-files NAME=FILE.")

    if "voting" in args.retrieval_modes and not args.num_votes:
        parser.error("Voting mode selected but no --num-votes provided.")

    return args


def iter_mode_vote_pairs() -> Iterable[Tuple[str, Optional[int]]]:
    """Yield (mode, num_votes) tuples respecting each retrieval mode's requirements."""
    for mode in RETRIEVAL_MODES:
        if mode == "voting":
            for votes in NUM_VOTES:
                yield mode, votes
        else:
            yield mode, None


def find_checkpoint_directories() -> Dict[str, str]:
    """Find the checkpoint directory for each experiment ID."""
    checkpoint_dirs: Dict[str, str] = {}
    for exp_id in EXPERIMENT_IDS:
        pattern = os.path.join(BASE_CHECKPOINT_DIR, f"{exp_id}")
        matching_dirs = glob.glob(pattern)
        if not matching_dirs:
            print(f"Warning: No checkpoint directory found for experiment {exp_id}")
            continue
        if len(matching_dirs) > 1:
            print(f"Warning: Multiple directories found for {exp_id}: {matching_dirs}")
            print(f"Using the first one: {matching_dirs[0]}")
        checkpoint_dirs[exp_id] = matching_dirs[0]
        print(f"Found checkpoint directory for {exp_id}: {matching_dirs[0]}")
    return checkpoint_dirs


def parse_checkpoint_filename(filename: str) -> Tuple[float, int]:
    """
    Parse checkpoint filename to extract val_loss and epoch.
    Returns (val_loss, epoch). If parsing fails, returns (inf, 0).
    """
    val_loss_match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", filename)
    epoch_match = re.search(r"epoch=([0-9]+)", filename)
    if not val_loss_match or not epoch_match:
        print(f"Warning: Could not parse filename {filename}")
        return float("inf"), 0
    return float(val_loss_match.group(1)), int(epoch_match.group(1))


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Select checkpoint with lowest val_loss, tiebreaker highest epoch."""
    checkpoint_files = os.listdir(checkpoint_dir)
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    best_checkpoint = None
    best_val_loss = float("inf")
    best_epoch = 0
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        val_loss, epoch = parse_checkpoint_filename(checkpoint_file)
        if (val_loss < best_val_loss) or (val_loss == best_val_loss and epoch > best_epoch):
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint = checkpoint_path
    print(f"Best checkpoint: {os.path.basename(best_checkpoint)} (val_loss={best_val_loss}, epoch={best_epoch})")
    return best_checkpoint


def build_common_cmd_args() -> List[str]:
    """Build dataset and device CLI overrides common to all runs."""
    args: List[str] = [
        f"runtime.device={DEVICE}",
    ]
    for dataset in DATASETS:
        if dataset not in DATASETS_DIRS or dataset not in DATASETS_FILES:
            continue
        args.append(f'data.{dataset}.path="{DATASETS_DIRS[dataset]}"')
        args.append(f'data.{dataset}.val="{DATASETS_FILES[dataset]}"')
    return args


def run_retrieval_evaluation(model_cfg: Dict[str, Any], mode: str, num_examples: int,
                             num_votes: Optional[int], run_tag: str, base_output_dir: str) -> Dict[str, Any]:
    """
    Run evaluate_retrieval.py with provided configuration.

    model_cfg: Dict containing keys understood by configs/evaluate_retrieval.yaml under model.
    mode: "1d_vectors" or "voting".
    num_examples: number of examples to evaluate.
    num_votes: used only for voting mode.
    run_tag: identifier to include in output directory naming.
    base_output_dir: parent output directory.
    Returns a dictionary with parsed metrics and paths, or error info.
    """
    if mode == "voting" and num_votes is None:
        raise ValueError("Voting mode requires num_votes to be specified.")

    votes_override = num_votes if num_votes is not None else NON_VOTING_PLACEHOLDER_VOTES
    run_output_dir = os.path.join(
        base_output_dir,
        f"{run_tag}_{mode}_N{num_examples}" + (f"_V{num_votes}" if mode == "voting" else "")
    )
    os.makedirs(run_output_dir, exist_ok=True)

    cmd = [
        "python", str(EVALUATE_RETRIEVAL_SCRIPT),
        f'model.checkpoint="{model_cfg["checkpoint"]}"',
        f'logging.output_dir="{run_output_dir}"',
        f"eval.num_examples={num_examples}",
        f"retrieval.mode={mode}",
        f"retrieval.mode_kwargs.num_votes={votes_override}",
    ]
    if TOPK_METRICS:
        topk_arg = ",".join(str(k) for k in TOPK_METRICS)
        cmd.append(f"retrieval.topk_metrics=[{topk_arg}]")

    # Optional model fields (for SALAD or other loaders)
    for opt_key in ("local_path", "salad_weights_path"):
        if opt_key in model_cfg and model_cfg[opt_key] is not None:
            cmd.append(f'model.{opt_key}="{model_cfg[opt_key]}"')

    # Common dataset and device args
    cmd.extend(build_common_cmd_args())

    votes_display = num_votes if mode == "voting" else "n/a"
    print(f"Running retrieval evaluation: tag={run_tag}, mode={mode}, N={num_examples}, votes={votes_display}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print("Error running retrieval evaluation:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return {"error": f"Evaluation failed: {result.stderr}", "output_dir": run_output_dir}

        metrics_path = os.path.join(run_output_dir, "retrieval_metrics.json")
        if not os.path.exists(metrics_path):
            return {"error": f"Metrics file not found: {metrics_path}", "output_dir": run_output_dir}

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Flatten summary of key metrics
        summary: Dict[str, Any] = {
            "tag": run_tag,
            "mode": mode,
            "num_examples": num_examples,
            "num_votes": (num_votes if mode == "voting" else None),
            "output_dir": run_output_dir,
        }
        mod0_metrics = metrics.get("mod0_to_mod1", {})
        summary["accuracy"] = mod0_metrics.get("accuracy")
        for k in TOPK_METRICS:
            summary[f"top{k}_accuracy"] = mod0_metrics.get(f"top{k}_accuracy")
        summary["avg_pred_vote_pct"] = mod0_metrics.get("avg_predicted_vote_percentage")
        summary["avg_correct_vote_pct"] = mod0_metrics.get("avg_correct_vote_percentage")
        summary["num_examples"] = mod0_metrics.get("num_examples")

        return summary
    except Exception as e:
        return {"error": f"Exception: {str(e)}", "output_dir": run_output_dir}


def main():
    args = parse_args()

    global EXPERIMENT_IDS, BASE_CHECKPOINT_DIR, BASE_OUTPUT_DIR, DATASETS, DATASETS_DIRS, DATASETS_FILES
    global RETRIEVAL_MODES, NUM_EXAMPLES, NUM_VOTES, TOPK_METRICS, DEVICE, EVALUATE_SALAD, SALAD_MODEL_CFG

    EXPERIMENT_IDS = list(args.experiment_ids)
    BASE_CHECKPOINT_DIR = args.base_checkpoint_dir
    BASE_OUTPUT_DIR = args.base_output_dir
    DATASETS = list(args.datasets)
    DATASETS_DIRS = dict(args.datasets_dirs)
    DATASETS_FILES = dict(args.datasets_files)
    RETRIEVAL_MODES = list(args.retrieval_modes)
    NUM_EXAMPLES = list(args.num_examples)
    NUM_VOTES = list(args.num_votes)
    TOPK_METRICS = sorted(set(int(k) for k in args.topk_metrics)) or TOPK_METRICS
    DEVICE = args.device
    EVALUATE_SALAD = args.evaluate_salad
    SALAD_MODEL_CFG = {
        "checkpoint": args.salad_checkpoint,
        "local_path": args.salad_local_path,
        "salad_weights_path": args.salad_weights_path,
    }

    print("Starting multi-run retrieval evaluation...")
    print(f"Experiment IDs: {EXPERIMENT_IDS}")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    # Optional SALAD evaluation
    if EVALUATE_SALAD:
        run_tag = "SALAD"
        for mode, votes in iter_mode_vote_pairs():
            for n in NUM_EXAMPLES:
                result = run_retrieval_evaluation(SALAD_MODEL_CFG, mode, n, votes, run_tag, BASE_OUTPUT_DIR)
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)

    # Contrastive checkpoints discovered by experiment IDs
    checkpoint_dirs = find_checkpoint_directories()
    for exp_id in EXPERIMENT_IDS:
        if exp_id not in checkpoint_dirs:
            print(f"Skipping {exp_id}: No checkpoint dir found")
            continue
        try:
            best_ckpt = find_best_checkpoint(checkpoint_dirs[exp_id])
        except Exception as e:
            print(f"Error finding best checkpoint for {exp_id}: {e}")
            continue

        model_cfg = {"checkpoint": best_ckpt}
        run_tag = f"exp_{exp_id}"
        for mode, votes in iter_mode_vote_pairs():
            for n in NUM_EXAMPLES:
                result = run_retrieval_evaluation(model_cfg, mode, n, votes, run_tag, BASE_OUTPUT_DIR)
                result["timestamp"] = datetime.now().isoformat()
                result["checkpoint"] = best_ckpt
                all_results.append(result)

    # Save combined CSV and JSON
    combined_csv = os.path.join(BASE_OUTPUT_DIR, "combined_retrieval_results.csv")
    combined_json = os.path.join(BASE_OUTPUT_DIR, "retrieval_summary.json")
    df = pd.DataFrame(all_results)
    df.to_csv(combined_csv, index=False)
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print compact summary table (successful runs only)
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)
    display_cols = ["tag", "mode", "num_examples", "num_votes", "accuracy"]
    display_cols.extend([f"top{k}_accuracy" for k in TOPK_METRICS if k != 1])
    display_cols.extend(["avg_pred_vote_pct", "avg_correct_vote_pct"])
    if not df.empty:
        success_df = df[df["error"].isna()] if "error" in df.columns else df
        if not success_df.empty:
            print(success_df[display_cols].to_string(index=False))
        else:
            print("No successful runs.")
    else:
        print("No runs executed.")

    print(f"\nResults saved to:\n  - Combined CSV: {combined_csv}\n  - JSON summary: {combined_json}\n  - Individual run outputs in: {BASE_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()




