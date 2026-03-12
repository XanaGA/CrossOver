#!/usr/bin/env python3
"""
Multi-checkpoint validation script

This script:
1) Finds the best checkpoint for each experiment ID (lowest val_loss, highest epoch on tie)
2) Runs validation for each checkpoint across easy, medium, hard difficulties
3) Saves results with clear naming for comparison

Usage:
    python scripts/run_multiple_val.py
"""

import os
import re
import subprocess
import glob
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import pandas as pd
import json
from datetime import datetime
import argparse

# Experiment IDs to validate
# EXPERIMENT_IDS = ["45119404", "44976966", "45119406", "46303147", "46303257"]
EXPERIMENT_IDS = ["45119404", "46303147", "46303257"]

# Base paths
BASE_CHECKPOINT_DIR = "outputs/contrastive/checkpoints/remote"
BASE_OUTPUT_DIR = "outputs/validation_results"

# TTA - list of TTA values to test
TTA = [1, 8]  # Use [1] for no TTA, [8] for 8 augmentations, or [1, 8] for both

# Datasets
# DATASETS = ["cubicasa5k", "structured3d"]
DATASETS = ["aria_synthenv"]
DATASETS_DIRS = {
    "cubicasa5k": "data/cubicasa5k",
    "structured3d": "data/structure3D/Structured3D_bbox/Structured3D",
    "aria_synthenv": "data/aria/SyntheticEnv/rendered_data",
    "swiss_dwellings": "data/SwissDwellings/modified-swiss-dwellings-v2/all_render",
    "zillow": "data/zind/rendered_data",
}

DATASETS_FILES = {
    "cubicasa5k": "data/cubicasa5k/val.txt",
    "structured3d": "data/structure3D/val.json",
    "aria_synthenv": "data/aria/SyntheticEnv/val.txt",
    "swiss_dwellings": "data/SwissDwellings/modified-swiss-dwellings-v2/val.txt",
    "zillow": "data/zind/val.txt",
}

# Difficulties
DIFFICULTIES = ["easy", "medium", "hard"]
# DIFFICULTIES = ["rot_only"]

# Validation loader settings
VAL_BATCH_SIZE = 32

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
    default_cwd = "/home/xavi/mmfe"
    default_batch_size = VAL_BATCH_SIZE
    parser = argparse.ArgumentParser(description="Run multi-checkpoint validation with configurable options.")

    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=default_experiment_ids,
        help="Experiment IDs to validate (space-separated).",
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
        help="Directory to store validation outputs.",
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
        help="Difficulties to evaluate (space-separated, e.g., easy medium hard).",
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
        help="Validation dataloader batch size.",
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
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        Path to the best checkpoint file
    """
    # checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    checkpoint_files = os.listdir(checkpoint_dir)
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
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


def run_validation(checkpoint_path: str, difficulty: str, exp_id: str, tta_value: int, cwd: str = "/home/xavi/mmfe") -> Dict[str, Any]:
    """
    Run validation for a specific checkpoint and difficulty.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        difficulty: "easy", "medium", "hard", or "rot_only"
        exp_id: Experiment ID
        tta_value: TTA value (1 for no TTA, >1 for TTA with rotations)
    
    Returns:
        Dictionary containing validation results
    """
    # Create output directory for this experiment
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"exp_{exp_id}_{difficulty}_tta{tta_value}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output CSV path
    output_csv = os.path.join(output_dir, "validation_metrics.csv")
    
    # Build validation command
    cmd = [
        sys.executable, "scripts/evaluation_scripts/validate.py",
        f'model.checkpoint="{checkpoint_path}"',  # Quote the path to handle brackets
        f"transforms.tf_difficulty={difficulty}",
        f'logging.output_csv="{output_csv}"',  # Quote this too for consistency
        "viz_debug=false",  # Disable debug visualization for batch runs
        f"tta={tta_value}"  # Number of augmentations (1 = no TTA, >1 = TTA with rotations)
    ]

    cmd.append(f"val.batch_size={VAL_BATCH_SIZE}")

    for dataset in DATASETS:
        cmd.append(f'data.{dataset}.path="{DATASETS_DIRS[dataset]}"')
        cmd.append(f'data.{dataset}.val="{DATASETS_FILES[dataset]}"')
    
    print(f"Running validation for {exp_id} ({difficulty})...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run validation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        
        if result.returncode != 0:
            print(f"Error running validation for {exp_id} ({difficulty}):")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return {"error": f"Validation failed: {result.stderr}"}
        
        # Load results
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            
            # Parse checkpoint metadata
            checkpoint_filename = os.path.basename(checkpoint_path)
            metadata = parse_checkpoint_metadata(checkpoint_filename)
            
            summary = {
                "experiment_id": exp_id,
                "difficulty": difficulty,
                "checkpoint": checkpoint_path,
                "common_rts": metadata["common_rts"],
                "noise_rts": metadata["noise_rts"],
                "train_eq": metadata["train_eq"],
                "tta": tta_value,
                "mean_val_loss": float(df["val_loss"].mean()),
                "std_val_loss": float(df["val_loss"].std()),
                "mean_val_acc_all": float(df["val_acc_all"].mean()),
                "std_val_acc_all": float(df["val_acc_all"].std()),
                "mean_val_acc_self": float(df["val_acc_self"].mean()),
                "std_val_acc_self": float(df["val_acc_self"].std()),
                "mean_val_acc_others": float(df["val_acc_others"].mean()),
                "std_val_acc_others": float(df["val_acc_others"].std()),
                "mean_val_acc_no_self": float(df["val_acc_no_self"].mean()),
                "std_val_acc_no_self": float(df["val_acc_no_self"].std()),
                "mean_tta_accuracy": float(df["tta_accuracy"].mean()),
                "std_tta_accuracy": float(df["tta_accuracy"].std()),
                "num_batches": len(df),
                "output_csv": output_csv
            }
            
            # Add Euclidean distance if it exists
            if "val_AEPE" in df.columns:
                summary["mean_val_AEPE"] = float(df["val_AEPE"].mean())
                summary["std_val_AEPE"] = float(df["val_AEPE"].std())
            
            # Add distance threshold metrics if they exist
            distance_threshold_columns = [col for col in df.columns if col.startswith("val_PCK@")]
            for col in distance_threshold_columns:
                summary[f"mean_{col}"] = float(df[col].mean())
                summary[f"std_{col}"] = float(df[col].std())
            
            # Add top-k metrics if they exist
            topk_columns = [col for col in df.columns if col.startswith("val_acc_top")]
            for col in topk_columns:
                summary[f"mean_{col}"] = float(df[col].mean())
                summary[f"std_{col}"] = float(df[col].std())
            
            print(f"Validation completed for {exp_id} ({difficulty}):")
            print(f"  Mean val_loss: {summary['mean_val_loss']:.6f} ± {summary['std_val_loss']:.6f}")
            print(f"  Mean val_acc_all: {summary['mean_val_acc_all']:.4f} ± {summary['std_val_acc_all']:.4f}")
            print(f"  Mean val_acc_self: {summary['mean_val_acc_self']:.4f} ± {summary['std_val_acc_self']:.4f}")
            print(f"  Mean val_acc_others: {summary['mean_val_acc_others']:.4f} ± {summary['std_val_acc_others']:.4f}")
            print(f"  Mean val_acc_no_self: {summary['mean_val_acc_no_self']:.4f} ± {summary['std_val_acc_no_self']:.4f}")
            print(f"  Mean tta_accuracy: {summary['mean_tta_accuracy']:.4f} ± {summary['std_tta_accuracy']:.4f}")
            
            # Print Euclidean distance if it exists
            if "mean_val_euclidean_distance" in summary:
                print(f"  Mean val_euclidean_distance: {summary['mean_val_euclidean_distance']:.4f} ± {summary['std_val_euclidean_distance']:.4f}")
            
            # Print distance threshold metrics if they exist
            for col in distance_threshold_columns:
                mean_key = f"mean_{col}"
                std_key = f"std_{col}"
                if mean_key in summary and std_key in summary:
                    print(f"  Mean {col}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")
            
            # Print top-k metrics if they exist
            for col in topk_columns:
                mean_key = f"mean_{col}"
                std_key = f"std_{col}"
                if mean_key in summary and std_key in summary:
                    print(f"  Mean {col}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")
            
            return summary
        else:
            return {"error": f"Output CSV not found: {output_csv}"}
            
    except Exception as e:
        print(f"Exception during validation for {exp_id} ({difficulty}): {e}")
        return {"error": f"Exception: {str(e)}"}

def main():
    """Main function to run validation for all experiments and difficulties."""
    args = parse_args()

    # Override module-level configuration from CLI args
    global EXPERIMENT_IDS, BASE_CHECKPOINT_DIR, BASE_OUTPUT_DIR, TTA, DATASETS, DATASETS_DIRS, DATASETS_FILES, DIFFICULTIES, VAL_BATCH_SIZE
    EXPERIMENT_IDS = list(args.experiment_ids)
    BASE_CHECKPOINT_DIR = args.base_checkpoint_dir
    BASE_OUTPUT_DIR = args.base_output_dir
    TTA = list(args.tta)
    DATASETS = list(args.datasets)
    DATASETS_DIRS = dict(args.datasets_dirs)
    DATASETS_FILES = dict(args.datasets_files)
    DIFFICULTIES = list(args.difficulties)
    VAL_BATCH_SIZE = int(args.batch_size)
    CWD = args.cwd  
    print("Starting multi-checkpoint validation...")
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
        
        # Run validation for each difficulty and TTA value
        for tta_value in TTA:
            for difficulty in DIFFICULTIES:
                print(f"\n{'-'*40}")
                print(f"Running validation: {exp_id} - {difficulty} - TTA={tta_value}")
                print(f"{'-'*40}")
                
                result = run_validation(best_checkpoint, difficulty, exp_id, tta_value, cwd=CWD)
                result["timestamp"] = datetime.now().isoformat()
                all_results.append(result)
    
    # Save combined results
    results_df = pd.DataFrame(all_results)
    combined_output = os.path.join(BASE_OUTPUT_DIR, "combined_validation_results.csv")
    results_df.to_csv(combined_output, index=False)
    
    # Save JSON summary for easy consumption
    json_output = os.path.join(BASE_OUTPUT_DIR, "validation_summary.json")
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if not all_results:
        print("No successful validations completed.")
        return
    
    # Create summary table
    summary_data = []
    for result in all_results:
        if "error" not in result:
            summary_data.append({
                "Experiment": result["experiment_id"],
                "Difficulty": result["difficulty"],
                "TTA": result["tta"],
                "Common RTS": result["common_rts"],
                "Noise RTS": result["noise_rts"],
                "Train Eq": result["train_eq"],
                "Val Loss": f"{result['mean_val_loss']:.6f} ± {result['std_val_loss']:.6f}",
                "Acc All": f"{result['mean_val_acc_all']:.4f} ± {result['std_val_acc_all']:.4f}",
                "TTA Accuracy": f"{result['mean_tta_accuracy']:.4f} ± {result['std_tta_accuracy']:.4f}",
                "Acc Self": f"{result['mean_val_acc_self']:.4f} ± {result['std_val_acc_self']:.4f}",
                "Acc Others": f"{result['mean_val_acc_others']:.4f} ± {result['std_val_acc_others']:.4f}",
                "Acc No Self": f"{result['mean_val_acc_no_self']:.4f} ± {result['std_val_acc_no_self']:.4f}",
                "Batches": result["num_batches"]
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
