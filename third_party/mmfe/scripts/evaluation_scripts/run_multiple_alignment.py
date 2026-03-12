#!/usr/bin/env python3
"""
Multi-checkpoint alignment evaluation script

This script:
1) Finds the best checkpoint for each experiment ID (lowest val_loss, highest epoch on tie)
2) Runs alignment evaluation for each checkpoint across easy, medium, hard difficulties
3) Saves results with clear naming for comparison

Usage:
    python scripts/evaluation_scripts/run_multiple_alignment.py
"""

import os
import re
import subprocess
import glob
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime
import argparse
import numpy as np

# Experiment IDs to evaluate
# EXPERIMENT_IDS = ["45119404", "46303147", "46303257"]
EXPERIMENT_IDS = ["45119404"]

# Base paths
BASE_CHECKPOINT_DIR = "outputs/contrastive/checkpoints/remote"
BASE_OUTPUT_DIR = "outputs/alignment_results"

# TTA - list of TTA values to test
TTA = [1, 8]  # Use [1] for no TTA, [8] for 8 augmentations, or [1, 8] for both

# Datasets
DATASETS = ["cubicasa", "structured3d"]
# DATASETS = ["aria_synthenv"]
DATASETS_DIRS = {
    "cubicasa": "data/cubicasa5k",
    "structured3d": "data/structure3D/Structured3D_bbox/Structured3D",
    "aria_synthenv": "data/aria/SyntheticEnv/rendered_data",
    "swiss_dwellings": "data/SwissDwellings/modified-swiss-dwellings-v2/all_render",
    "zillow": "data/zind/rendered_data",
}

DATASETS_FILES = {
    "cubicasa": "data/cubicasa5k/val.txt",
    "structured3d": "data/structure3D/val.json",
    "aria_synthenv": "data/aria/SyntheticEnv/val.txt",
    "swiss_dwellings": "data/SwissDwellings/modified-swiss-dwellings-v2/val.txt",
    "zillow": "data/zind/val.txt",
}

# Difficulties
DIFFICULTIES = ["medium", "hard"]
# DIFFICULTIES = ["rot_only"]

# Evaluation methods
EVAL_METHODS = ["ransac"]  # Options: "ransac", "lmeds", "best"

# Alignment threshold (pixels)
ALIGNMENT_THRESHOLD = [3, 5, 10]

# Filter by certainty for RoMa TTA
FILTER_BY_CERTAINTY = True  # If False, estimate affine for each augmentation and pick best by inliers

# Evaluation batch size
EVAL_BATCH_SIZE = 32

# Optional RoMa->MMFE backbone checkpoint override
ROMA_MMFE_CHECKPOINT = None

def parse_key_value_list(pairs: List[str]) -> Dict[str, str]:
    """Parse a list of KEY=VALUE strings into a dictionary."""
    result: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE format, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Empty key in pair: {pair}")
        result[key] = value
    return result

def parse_args() -> argparse.Namespace:
    """Parse command line arguments to override defaults."""
    # Defaults derived from current module-level values
    default_experiment_ids = EXPERIMENT_IDS
    default_base_checkpoint_dir = BASE_CHECKPOINT_DIR
    default_base_output_dir = BASE_OUTPUT_DIR
    default_tta = TTA
    default_datasets = DATASETS
    default_datasets_dirs_pairs = [f"{k}={v}" for k, v in DATASETS_DIRS.items()]
    default_datasets_files_pairs = [f"{k}={v}" for k, v in DATASETS_FILES.items()]
    default_difficulties = DIFFICULTIES
    default_eval_methods = EVAL_METHODS
    default_alignment_threshold = ALIGNMENT_THRESHOLD
    default_filter_by_certainty = FILTER_BY_CERTAINTY
    default_batch_size = EVAL_BATCH_SIZE
    default_cwd = "/home/xavi/mmfe"
    default_use_dino_res = False
    default_upsampler_output_size = None
    default_roma_mmfe_checkpoint = ROMA_MMFE_CHECKPOINT
    parser = argparse.ArgumentParser(description="Run multi-checkpoint alignment evaluation with configurable options.")

    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=default_experiment_ids,
        help="Experiment IDs to evaluate (space-separated).",
    )

    parser.add_argument(
        "--base-checkpoint-dir",
        type=str,
        default=default_base_checkpoint_dir,
        help="Base directory containing experiment checkpoint subdirectories.",
    )

    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=default_base_output_dir,
        help="Directory to store alignment results.",
    )

    parser.add_argument(
        "--tta",
        nargs="+",
        type=int,
        default=default_tta,
        help="TTA values to evaluate (space-separated integers).",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=default_datasets,
        help="Datasets to evaluate (space-separated).",
    )

    parser.add_argument(
        "--datasets-dirs",
        nargs="+",
        default=default_datasets_dirs_pairs,
        metavar="NAME=PATH",
        help="Mapping of dataset name to data directory (repeat as NAME=PATH).",
    )

    parser.add_argument(
        "--datasets-files",
        nargs="+",
        default=default_datasets_files_pairs,
        metavar="NAME=FILE",
        help="Mapping of dataset name to validation file (repeat as NAME=FILE).",
    )

    parser.add_argument(
        "--difficulties",
        nargs="+",
        type=str,
        default=default_difficulties,
        help="Difficulties to evaluate (space-separated).",
    )

    parser.add_argument(
        "--eval-methods",
        nargs="+",
        type=str,
        default=default_eval_methods,
        help='Evaluation methods (e.g., ransac lmeds best).',
    )

    parser.add_argument(
        "--alignment-threshold",
        nargs="+",
        type=int,
        default=default_alignment_threshold,
        help="Alignment thresholds (pixels) for accuracy computation (space-separated integers).",
    )

    parser.add_argument(
        "--filter-by-certainty",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=default_filter_by_certainty,
        help="Filter matches by certainty (True) or estimate affine for each augmentation and pick best by inliers (False). Default: True",
    )

    parser.add_argument(
        "--cwd",
        type=str,
        default=default_cwd,
        help="Current working directory.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="Batch size to use for alignment evaluation.",
    )

    parser.add_argument(
        "--use-dino-res",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=default_use_dino_res,
        help="Use DINO ResNet features (True) or not (False). Default: False",
    )

    parser.add_argument(
        "--upsampler-output-size",
        nargs="+",
        type=int,
        default=default_upsampler_output_size,
        help="Upsampler output size (width, height) (space-separated integers).",
    )

    parser.add_argument(
        "--roma-mmfe-checkpoint",
        type=str,
        default=default_roma_mmfe_checkpoint,
        help="Optional path to an MMFE checkpoint to use as the RoMa backbone.",
    )

    args = parser.parse_args()

    # Normalize and validate mappings
    args.datasets_dirs = parse_key_value_list(args.datasets_dirs)
    args.datasets_files = parse_key_value_list(args.datasets_files)

    # Ensure provided datasets exist in mappings
    missing_dir = [d for d in args.datasets if d not in args.datasets_dirs]
    missing_file = [d for d in args.datasets if d not in args.datasets_files]
    if missing_dir:
        parser.error(f"Missing dataset directories for: {missing_dir}. Provide with --datasets-dirs NAME=PATH")
    if missing_file:
        parser.error(f"Missing dataset files for: {missing_file}. Provide with --datasets-files NAME=FILE")

    return args


def find_checkpoint_directories() -> Dict[str, str]:
    """Find the checkpoint directory for each experiment ID."""
    checkpoint_dirs = {}
    
    for exp_id in EXPERIMENT_IDS:
        # Special case: allow using the ROMA matcher directly
        if exp_id == "roma_v2" or exp_id == "roma_v1":
            checkpoint_dirs[exp_id] = str(exp_id).lower()
            print(f"Using ROMA matcher for experiment id '{exp_id}' (no checkpoint directory needed)")
            continue
        if "dinov3_vitb16" in str(exp_id).lower():
            checkpoint_dirs[exp_id] = str(exp_id).lower()
            print(f"Using DINOv3 ViT-B16 matcher for experiment id '{exp_id}' (no checkpoint directory needed)")
            continue
        if "dinov2_vitb14" in str(exp_id).lower():
            checkpoint_dirs[exp_id] = str(exp_id).lower()
            print(f"Using DINOv2 ViT-B14 matcher for experiment id '{exp_id}' (no checkpoint directory needed)")
            continue
        # Search for directories matching the pattern
        pattern = os.path.join(BASE_CHECKPOINT_DIR, f"{exp_id}*")
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
    
    Args:
        filename: Checkpoint filename (e.g., "44557240_...-val_loss=0.0022-epoch=33.ckpt")
    
    Returns:
        Tuple of (val_loss, epoch)
    """
    # Extract val_loss and epoch from filename
    # Use more precise regex to avoid capturing trailing dots
    val_loss_match = re.search(r'val_loss=([0-9]+(?:\.[0-9]+)?)', filename)
    epoch_match = re.search(r'epoch=([0-9]+)', filename)
    
    if not val_loss_match or not epoch_match:
        print(f"Warning: Could not parse filename {filename}")
        print(f"  val_loss_match: {val_loss_match}")
        print(f"  epoch_match: {epoch_match}")
        return float('inf'), 0
    
    val_loss = float(val_loss_match.group(1))
    epoch = int(epoch_match.group(1))
    
    return val_loss, epoch


def parse_checkpoint_metadata(filename: str) -> Dict[str, str]:
    """
    Parse checkpoint filename to extract training configuration metadata.
    
    Args:
        filename: Checkpoint filename (e.g., "44375049_proj_dim_32_dino_MoGe1_common_180_[0.0,0.0]_[1.0,1.0]_noise_0_[0.0,0.0]_[1.0,1.0]_train_eq_false-epoch=29-val_loss=0.0018.ckpt")
    
    Returns:
        Dictionary with common_rts, noise_rts, and train_eq
    """
    # Special case: if filename is "last.ckpt", return unknown for all metadata
    if filename == "last.ckpt":
        return {
            "common_rts": "unknown",
            "noise_rts": "unknown",
            "train_eq": "unknown"
        }
    
    # Extract common_rts: everything between "common_" and "_noise_"
    common_match = re.search(r'common_([^_]+_[^_]+_[^_]+)_noise_', filename)
    
    # Extract noise_rts: everything between "_noise_" and "_train_eq_"
    noise_match = re.search(r'_noise_([^_]+_[^_]+_[^_]+)_train_eq_', filename)
    
    # Extract train_eq: everything between "_train_eq_" and "-epoch"
    train_eq_match = re.search(r'_train_eq_([^-]+)-epoch', filename)
    
    metadata = {}
    
    if common_match:
        metadata["common_rts"] = common_match.group(1)
    else:
        metadata["common_rts"] = "unknown"
        
    if noise_match:
        metadata["noise_rts"] = noise_match.group(1)
    else:
        metadata["noise_rts"] = "unknown"
        
    if train_eq_match:
        metadata["train_eq"] = train_eq_match.group(1)
    else:
        metadata["train_eq"] = "unknown"
    
    return metadata


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the best checkpoint in a directory.
    
    Selection criteria:
    1. Lowest val_loss
    2. If tied, highest epoch
    
    Special case: If there's only one checkpoint called "last.ckpt", use it directly.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        Path to the best checkpoint file
    """
    # If using ROMA directly, the "checkpoint" is just the literal string
    if checkpoint_dir == "roma_v2" or checkpoint_dir == "roma_v1" or checkpoint_dir == "dinov3_vitb16" or checkpoint_dir == "dinov2_vitb14":
        return checkpoint_dir
    # checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    all_files = os.listdir(checkpoint_dir)
    # Filter for .ckpt files only
    checkpoint_files = [f for f in all_files if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Special case: if there's only one checkpoint and it's "last.ckpt", use it directly
    if len(checkpoint_files) == 1 and checkpoint_files[0] == "last.ckpt":
        best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
        print(f"Using single checkpoint: {checkpoint_files[0]}")
        return best_checkpoint
    
    best_checkpoint = None
    best_val_loss = float('inf')
    best_epoch = 0
    
    for checkpoint_file in checkpoint_files:
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)
        filename = os.path.basename(checkpoint_file)
        val_loss, epoch = parse_checkpoint_filename(filename)
        
        # Selection logic: lower val_loss, or if tied, higher epoch
        if (val_loss < best_val_loss) or (val_loss == best_val_loss and epoch > best_epoch):
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint = checkpoint_file
    
    print(f"Best checkpoint: {os.path.basename(best_checkpoint)} (val_loss={best_val_loss}, epoch={best_epoch})")
    return best_checkpoint


def run_alignment_evaluation(checkpoint_path: str, difficulty: str, exp_id: str, 
                             tta_value: int, method: str = "ransac", filter_by_certainty: bool = True,
                             cwd: str = "/home/xavi/mmfe", use_dino_res: bool = False,
                             upsampler_output_size: tuple = (32, 32), roma_mmfe_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Run alignment evaluation for a specific checkpoint and difficulty.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        difficulty: "easy", "medium", "hard", or "rot_only"
        exp_id: Experiment ID
        tta_value: TTA value (1 for no TTA, >1 for TTA with rotations)
        method: Affine matrix estimation method ("ransac", "lmeds", "best")
        filter_by_certainty: If True, filter matches by certainty. If False, estimate affine for each augmentation and pick best by inliers.
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory for this experiment
    filter_suffix = "filtered" if filter_by_certainty else "inliers"
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"exp_{exp_id}_{difficulty}_tta{tta_value}_{method}_{filter_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output CSV path
    output_csv = os.path.join(output_dir, "alignment_metrics.csv")
    
    # Build evaluation command
    cmd = [
        sys.executable, "scripts/evaluation_scripts/evaluate_aligment.py",
        f'model.checkpoint="{checkpoint_path}"',  # Quote the path to handle brackets
        f"transforms.tf_difficulty={difficulty}",
        f'logging.output_csv="{output_csv}"',  # Quote this too for consistency
        f"eval.tta_n_augs={tta_value}",  # Number of augmentations (1 = no TTA, >1 = TTA with rotations)
        f"eval.method={method}",  # Affine matrix estimation method
        f"eval.threshold={ALIGNMENT_THRESHOLD}",  # Distance threshold for corner alignment
        f"eval.filter_by_certainty={filter_by_certainty}",  # Filter by certainty or use inliers
        f"model.kwargs.use_dino_res={use_dino_res}",
    ]
    if upsampler_output_size != None:
        cmd.append(f"model.kwargs.upsampler_output_size={upsampler_output_size}")
    cmd.append(f"eval.batch_size={EVAL_BATCH_SIZE}")
    if roma_mmfe_checkpoint:
        cmd.append(f'model.kwargs.roma_mmfe_checkpoint="{roma_mmfe_checkpoint}"')

    for dataset in DATASETS:
        cmd.append(f'data.{dataset}.path="{DATASETS_DIRS[dataset]}"')
        cmd.append(f'data.{dataset}.val="{DATASETS_FILES[dataset]}"')
    
    print(f"Running alignment evaluation for {exp_id} ({difficulty}, TTA={tta_value}, method={method}, filter_by_certainty={filter_by_certainty})...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
        
        if result.returncode != 0:
            print(f"Error running alignment evaluation for {exp_id} ({difficulty}):")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return {"error": f"Alignment evaluation failed: {result.stderr}"}
        
        # Load results
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            
            # Parse checkpoint metadata
            checkpoint_filename = os.path.basename(checkpoint_path)
            metadata = parse_checkpoint_metadata(checkpoint_filename)
            
            # Find accuracy columns (format: accuracy@{threshold})
            accuracy_cols = [col for col in df.columns if col.startswith('accuracy@')]
            
            if not accuracy_cols:
                print(f"Warning: No accuracy columns found for {exp_id} ({difficulty})")
                return {
                    "error": "No accuracy columns found",
                    "experiment_id": exp_id,
                    "difficulty": difficulty,
                    "checkpoint": checkpoint_path,
                    "tta": tta_value,
                    "method": method
                }
            
            # Filter valid results (non-NaN for any accuracy column)
            valid_df = df.dropna(subset=accuracy_cols)
            
            if len(valid_df) == 0:
                print(f"Warning: No valid results for {exp_id} ({difficulty})")
                return {
                    "error": "No valid results",
                    "experiment_id": exp_id,
                    "difficulty": difficulty,
                    "checkpoint": checkpoint_path,
                    "tta": tta_value,
                    "method": method
                }
            
            summary = {
                "experiment_id": exp_id,
                "difficulty": difficulty,
                "checkpoint": checkpoint_path,
                "common_rts": metadata["common_rts"],
                "noise_rts": metadata["noise_rts"],
                "train_eq": metadata["train_eq"],
                "tta": tta_value,
                "method": method,
                "filter_by_certainty": filter_by_certainty,
                "threshold": ALIGNMENT_THRESHOLD,
                "total_samples": len(df),
                "valid_samples": len(valid_df),
                "mean_rotation_error": float(valid_df["rotation_error"].mean()),
                "std_rotation_error": float(valid_df["rotation_error"].std()),
                "mean_distance": float(valid_df["mean_distance"].mean()),
                "std_distance": float(valid_df["mean_distance"].std()),
                "mean_rms_error": float(valid_df["rms_error"].mean()),
                "median_rms_error": float(np.median(valid_df["rms_error"])),
                "std_rms_error": float(valid_df["rms_error"].std()),
                "mean_median_error": float(valid_df["median_error"].mean()),
                "std_median_error": float(valid_df["median_error"].std()),
                "output_csv": output_csv
            }
            
            # Add accuracy statistics for each threshold
            for col in accuracy_cols:
                threshold_val = col.replace('accuracy@', '')
                summary[f"mean_{col}"] = float(valid_df[col].mean())
                summary[f"std_{col}"] = float(valid_df[col].std())
                summary[f"perfect_{col}"] = int((valid_df[col] == 100).sum())
                summary[f"perfect_{col}_pct"] = float((valid_df[col] == 100).sum() / len(valid_df) * 100)
            
            # Calculate accuracy distribution for each threshold
            for col in accuracy_cols:
                threshold_val = col.replace('accuracy@', '')
                for low, high in [(0, 25), (25, 50), (50, 75), (75, 100)]:
                    count = len(valid_df[(valid_df[col] >= low) & (valid_df[col] < high)])
                    summary[f"{col}_{low}_{high}"] = int(count)
                    summary[f"{col}_{low}_{high}_pct"] = float(count / len(valid_df) * 100)
            
            print(f"Alignment evaluation completed for {exp_id} ({difficulty}, TTA={tta_value}, method={method}, filter_by_certainty={filter_by_certainty}):")
            print(f"  Valid samples: {summary['valid_samples']}/{summary['total_samples']}")
            
            # Print accuracy for each threshold
            for col in accuracy_cols:
                threshold_val = col.replace('accuracy@', '')
                print(f"  Mean accuracy@{threshold_val}: {summary[f'mean_{col}']:.2f}% ± {summary[f'std_{col}']:.2f}%")
                print(f"  Perfect alignments@{threshold_val}: {summary[f'perfect_{col}']} ({summary[f'perfect_{col}_pct']:.1f}%)")
            
            print(f"  Mean rotation error: {summary['mean_rotation_error']:.2f}° ± {summary['std_rotation_error']:.2f}°")
            print(f"  Mean corner distance: {summary['mean_distance']:.2f} ± {summary['std_distance']:.2f} pixels")
            print(f"  Mean RMS error: {summary['mean_rms_error']:.2f} ± {summary['std_rms_error']:.2f} pixels")
            print(f"  Median RMS error: {summary['median_rms_error']:.2f} pixels")
            print(f"  Mean median error: {summary['mean_median_error']:.2f} ± {summary['std_median_error']:.2f} pixels")
            
            return summary
        else:
            return {"error": f"Output CSV not found: {output_csv}"}
            
    except Exception as e:
        print(f"Exception during alignment evaluation for {exp_id} ({difficulty}): {e}")
        return {"error": f"Exception: {str(e)}"}


def main():
    """Main function to run alignment evaluation for all experiments and difficulties."""
    args = parse_args()

    # Override module-level configuration from CLI args
    global EXPERIMENT_IDS, BASE_CHECKPOINT_DIR, BASE_OUTPUT_DIR, TTA, DATASETS, DATASETS_DIRS, DATASETS_FILES, DIFFICULTIES, EVAL_METHODS, ALIGNMENT_THRESHOLD, FILTER_BY_CERTAINTY, EVAL_BATCH_SIZE, ROMA_MMFE_CHECKPOINT
    EXPERIMENT_IDS = list(args.experiment_ids)
    BASE_CHECKPOINT_DIR = args.base_checkpoint_dir
    BASE_OUTPUT_DIR = args.base_output_dir
    TTA = list(args.tta)
    DATASETS = list(args.datasets)
    DATASETS_DIRS = dict(args.datasets_dirs)
    DATASETS_FILES = dict(args.datasets_files)
    DIFFICULTIES = list(args.difficulties)
    EVAL_METHODS = list(args.eval_methods)
    ALIGNMENT_THRESHOLD = list(args.alignment_threshold)
    FILTER_BY_CERTAINTY = args.filter_by_certainty
    EVAL_BATCH_SIZE = int(args.batch_size)
    CWD = args.cwd
    USE_DINO_RES = args.use_dino_res
    UPSAMPLER_OUTPUT_SIZE = args.upsampler_output_size
    ROMA_MMFE_CHECKPOINT = args.roma_mmfe_checkpoint
    print("Starting multi-checkpoint alignment evaluation...")
    print(f"Experiment IDs: {EXPERIMENT_IDS}")
    
    # Find checkpoint directories
    checkpoint_dirs = find_checkpoint_directories()
    
    if not checkpoint_dirs:
        print("No checkpoint directories found. Exiting.")
        return
    
    # Create main output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Process each experiment
    for exp_id in EXPERIMENT_IDS:
        if exp_id not in checkpoint_dirs:
            print(f"Skipping {exp_id}: No checkpoint directory found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing experiment {exp_id}")
        print(f"{'='*60}")
        
        # Find best checkpoint for this experiment
        try:
            best_checkpoint = find_best_checkpoint(checkpoint_dirs[exp_id])
        except Exception as e:
            print(f"Error finding best checkpoint for {exp_id}: {e}")
            continue
        
        # Run alignment evaluation for each difficulty, TTA value, and method
        for tta_value in TTA:
            for difficulty in DIFFICULTIES:
                for method in EVAL_METHODS:
                    print(f"\n{'-'*40}")
                    print(f"Running evaluation: {exp_id} - {difficulty} - TTA={tta_value} - method={method} - filter_by_certainty={FILTER_BY_CERTAINTY}")
                    print(f"{'-'*40}")
                    
                    result = run_alignment_evaluation(best_checkpoint, difficulty, exp_id, tta_value, method, 
                                                     filter_by_certainty=FILTER_BY_CERTAINTY, cwd=CWD, use_dino_res=USE_DINO_RES, upsampler_output_size=UPSAMPLER_OUTPUT_SIZE, roma_mmfe_checkpoint=ROMA_MMFE_CHECKPOINT)
                    result["timestamp"] = datetime.now().isoformat()
                    all_results.append(result)
    
    # Save combined results
    results_df = pd.DataFrame(all_results)
    combined_output = os.path.join(BASE_OUTPUT_DIR, "combined_alignment_results.csv")
    results_df.to_csv(combined_output, index=False)
    
    # Save JSON summary for easy consumption
    json_output = os.path.join(BASE_OUTPUT_DIR, "alignment_summary.json")
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ALIGNMENT EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if not all_results:
        print("No successful evaluations completed.")
        return
    
    # Create summary table
    summary_data = []
    for result in all_results:
        if "error" not in result:
            # Find accuracy columns in this result
            accuracy_cols = [key for key in result.keys() if key.startswith('mean_accuracy@')]
            
            # Create base row
            row = {
                "Experiment": result["experiment_id"],
                "Difficulty": result["difficulty"],
                "TTA": result["tta"],
                "Method": result["method"],
                "Filter By Cert": result.get("filter_by_certainty", "N/A"),
                "Common RTS": result["common_rts"],
                "Noise RTS": result["noise_rts"],
                "Train Eq": result["train_eq"],
                "Valid/Total": f"{result['valid_samples']}/{result['total_samples']}",
                "Rot Error": f"{result['mean_rotation_error']:.2f}° ± {result['std_rotation_error']:.2f}°",
                "Distance": f"{result['mean_distance']:.2f} ± {result['std_distance']:.2f}",
                "RMS": f"{result['mean_rms_error']:.2f} ± {result['std_rms_error']:.2f}",
                "Median RMS": f"{result['median_rms_error']:.2f}",
                "Median": f"{result['mean_median_error']:.2f} ± {result['std_median_error']:.2f}"
            }
            
            # Add accuracy columns for each threshold
            for col in accuracy_cols:
                threshold_val = col.replace('mean_accuracy@', '')
                std_col = f"std_accuracy@{threshold_val}"
                perfect_col = f"perfect_accuracy@{threshold_val}_pct"
                
                row[f"Acc@{threshold_val}"] = f"{result[col]:.2f} ± {result[std_col]:.2f}"
                row[f"Perfect@{threshold_val}"] = f"{result[perfect_col]:.1f}%"
            
            summary_data.append(row)
        else:
            summary_data.append({
                "Experiment": result.get("experiment_id", "N/A"),
                "Difficulty": result.get("difficulty", "N/A"),
                "TTA": result.get("tta", "N/A"),
                "Method": result.get("method", "N/A"),
                "Filter By Cert": result.get("filter_by_certainty", "N/A"),
                "Common RTS": "N/A",
                "Noise RTS": "N/A",
                "Train Eq": "N/A",
                "Valid/Total": "ERROR",
                "Rot Error": "-",
                "Distance": "-",
                "RMS": "-"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"  - Combined CSV: {combined_output}")
    print(f"  - JSON summary: {json_output}")
    print(f"  - Individual results in: {BASE_OUTPUT_DIR}/exp_*/")
    
    # Count successful vs failed runs
    successful = sum(1 for r in all_results if "error" not in r)
    failed = sum(1 for r in all_results if "error" in r)
    print(f"\nRun summary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()

