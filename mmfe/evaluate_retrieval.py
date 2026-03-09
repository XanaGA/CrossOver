#!/usr/bin/env python3
"""
Test cross-modal retrieval capabilities of a contrastive learning model.

This script:
- Loads a model checkpoint and validation dataset
- Computes embeddings for N examples from both modalities (2*N images total)
- For each example, retrieves the corresponding example in the other modality
- Uses voting-based retrieval: sample M cells from query embedding, find nearest neighbors,
  and vote for the most similar embedding
- Computes retrieval accuracy and voting statistics

Run example:
  python scripts/evaluation_scripts/evaluate_retrieval.py \
    model.checkpoint=/abs/path/model.ckpt \
    data.cubicasa.path=/abs/cubicasa5k data.cubicasa.val=/abs/cubicasa5k/val.txt \
    data.structured3d.path=/abs/Structured3D data.structured3d.val=/abs/Structured3D/val.json \
    data.image_size='[256,256]' \
    eval.num_examples=100 eval.num_votes=50 \
    logging.output_dir=outputs/metrics/retrieval
"""

import os
import sys
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from dotenv import load_dotenv

from global_descriptors.global_descriptor_model_aug import create_aug_global_descriptor_model
from global_descriptors.global_descriptor_modules import create_global_descriptor_model, create_salad_og_model
from global_descriptors.global_descriptor_models import GlobalDescriptorModel, load_global_descriptor_model_from_checkpoint

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import PairRandomAffine, PairToPIL, PairResize, PairGrayscale, PairToTensor, PairNormalize
from training.lightning_module import ContrastiveLearningModule
from mmfe_utils.data_utils import create_datasets
from mmfe_utils.viz_retrieval import save_retrieval_visualizations


def compute_all_embeddings(model, dataloader: DataLoader, num_examples: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for all examples in both modalities.
    
    Args:
        model: The contrastive learning model
        dataloader: DataLoader for the dataset
        num_examples: Number of examples to process
        device: Device to use for computation
        
    Returns:
        embeddings_0: Tensor of shape (N, C, H, W) for modality 0
        embeddings_1: Tensor of shape (N, C, H, W) for modality 1
    """
    embeddings_0_list = []
    embeddings_1_list = []
    queries_and_database = []
    
    batch_size = dataloader.batch_size
    num_batches = num_examples // batch_size
    if num_examples % batch_size != 0:
        num_batches += 1
    print(f"Computing embeddings for {num_batches*batch_size} examples (closest to {num_examples})...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches:
                break
                
            image0 = batch["modality_0"].to(device)
            image1 = batch["original_modality_1"].to(device)
            
            if isinstance(model, GlobalDescriptorModel):
                # Get embeddings (batch_size, C, H, W)
                e0, e1 = model.get_embeddings(image0, image1)
            else:
                e0 = model(image0)
                e1 = model(image1)
            
            embeddings_0_list.append(e0.cpu())
            embeddings_1_list.append(e1.cpu())
            queries_and_database += [{
                                    'modality_0': image0[b].cpu(),
                                    'original_modality_0': batch['original_modality_0'][b].cpu(),
                                    'original_modality_1': image1[b].cpu()
                                    } for b in range(len(image0))]
    
    # Concatenate all embeddings (N, C, H, W)
    embeddings_0 = torch.cat(embeddings_0_list, dim=0)
    embeddings_1 = torch.cat(embeddings_1_list, dim=0)

    print(f"Embeddings computed: modality_0 shape={embeddings_0.shape}, modality_1 shape={embeddings_1.shape}")
    return embeddings_0, embeddings_1, queries_and_database


def sample_random_cells(embedding: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Randomly sample M cells from an embedding grid.
    
    Args:
        embedding: Embedding tensor of shape (C, H, W)
        num_samples: Number of cells to sample
        
    Returns:
        sampled_features: Tensor of shape (M, C) with sampled feature vectors
        sampled_coords: Tensor of shape (M, 2) with (h, w) coordinates
    """
    C, H, W = embedding.shape
    
    # Flatten spatial dimensions
    embedding_flat = embedding.view(C, -1).transpose(0, 1)  # (H*W, C)
    
    # Random sampling
    total_cells = H * W
    indices = torch.randperm(total_cells)[:num_samples]
    
    sampled_features = embedding_flat[indices]  # (M, C)
    
    # Get coordinates for reference (not used in retrieval but useful for debugging)
    h_coords = indices // W
    w_coords = indices % W
    sampled_coords = torch.stack([h_coords, w_coords], dim=1)  # (M, 2)
    
    return sampled_features, sampled_coords


def find_nearest_embedding(query_cells: torch.Tensor, all_embeddings: torch.Tensor, 
                           exclude_idx: int = None) -> torch.Tensor:
    """
    Find the embedding with the nearest cell to each query cell using cosine similarity.
    Fully vectorized version for efficiency - processes multiple query cells in parallel.
    
    Args:
        query_cells: Feature vectors of shape (M, C) or (C,) for single query
        all_embeddings: Tensor of shape (N, C, H, W) containing all embeddings
        exclude_idx: Index to exclude from search 
        
    Returns:
        Tensor of indices of shape (M,) or scalar if input was (C,)
    """
    # Handle single query cell case
    return_scalar = False
    if query_cells.ndim == 1:
        query_cells = query_cells.unsqueeze(0)  # (1, C)
        return_scalar = True
    
    M, C = query_cells.shape
    N, C, H, W = all_embeddings.shape
    
    # Normalize query cells: (M, C)
    query_cells_norm = F.normalize(query_cells, dim=1)
    
    # Reshape embeddings: (N, C, H, W) -> (N, H*W, C) -> (N*H*W, C)
    embeddings_flat = all_embeddings.view(N, C, -1).transpose(1, 2)  # (N, H*W, C)
    embeddings_norm = F.normalize(embeddings_flat, dim=2)  # (N, H*W, C)
    embeddings_norm_flat = embeddings_norm.reshape(-1, C)  # (N*H*W, C)
    
    # Compute all similarities at once: (M, C) @ (C, N*H*W) -> (M, N*H*W)
    similarities = torch.matmul(query_cells_norm, embeddings_norm_flat.T)
    
    # Reshape to (M, N, H*W) to separate embeddings
    similarities = similarities.view(M, N, H * W)
    
    # Get maximum similarity for each embedding: (M, N)
    max_sims, _ = similarities.max(dim=2)
    
    # Exclude the query embedding itself if specified
    if exclude_idx is not None:
        max_sims[:, exclude_idx] = -float('inf')
    
    # Get best embedding for each query cell: (M,)
    best_indices = max_sims.argmax(dim=1)
    
    # Return scalar if input was single query cell
    if return_scalar:
        return best_indices.item()
    
    return best_indices


def retrieve_with_voting(query_embedding: torch.Tensor, target_embeddings: torch.Tensor,
                        query_idx: int, num_votes: int) -> Dict:
    """
    Retrieve the most similar embedding using voting-based approach.
    Fully vectorized - processes all votes in parallel.
    
    Args:
        query_embedding: Query embedding of shape (C, H, W)
        target_embeddings: Tensor of all embeddings in the target modality, shape (N, C, H, W)
        query_idx: Index of the query in its own modality (to compute correct match)
        num_votes: Number of cells to sample for voting
        
    Returns:
        Dictionary with retrieval results
    """
    # Sample random cells from query embedding
    sampled_features, _ = sample_random_cells(query_embedding, num_votes)  # (M, C)
    
    # Get votes for all sampled cells in parallel (no loop!)
    votes = find_nearest_embedding(sampled_features, target_embeddings, exclude_idx=None)  # (M,)
    
    # Convert to list for Counter
    votes_list = votes.tolist()
    
    # Count votes
    vote_counter = Counter(votes_list)
    predicted_idx = vote_counter.most_common(1)[0][0]
    predicted_votes = vote_counter[predicted_idx]
    correct_votes = vote_counter.get(query_idx, 0)
    
    return {
        'type': 'voting',
        'predicted_idx': predicted_idx,
        'correct_idx': query_idx,
        'predicted_votes': predicted_votes,
        'correct_votes': correct_votes,
        'total_votes': num_votes,
        'is_correct': predicted_idx == query_idx,
        'vote_distribution': dict(vote_counter)
    }

def retrieve_1d_vectors(embeddings_0: torch.Tensor, embeddings_1: torch.Tensor, max_topk: int = 10):
    """
    Retrieve the most similar embedding between two set of embeddings using cosine similarity.
    
    Args:
        embeddings_0: Embeddings for modality 0, shape (N, C)
        embeddings_1: Embeddings for modality 1, shape (N, C)
        
    Returns:
        Dictionary with retrieval results including top-k predictions
    """
    # Compute cosine similarities
    emb_0_norm = F.normalize(embeddings_0, dim=1)
    emb_1_norm = F.normalize(embeddings_1, dim=1)

    similarities = torch.matmul(emb_0_norm, emb_1_norm.T)
    
    # Get top-k predictions for each query (bounded by feature count)
    topk = min(max_topk, similarities.shape[1])
    _, topk_indices = torch.topk(similarities, k=topk, dim=1)
    
    return {
        "predicted_idx": similarities.argmax(dim=1),
        "correct_idx": torch.arange(embeddings_0.shape[0]).to(embeddings_0.device),
        "topk_indices": topk_indices,
    }

def evaluate_retrieval(embeddings_0: torch.Tensor, embeddings_1: torch.Tensor,
                       mode: str = 'voting', mode_kwargs: Dict = {}, topk_metrics: List[int] = None) -> Dict:
    """
    Evaluate cross-modal retrieval performance.
    
    Args:
        embeddings_0: Embeddings for modality 0, shape (N, C, H, W)
        embeddings_1: Embeddings for modality 1, shape (N, C, H, W)
        mode: Retrieval mode ('voting', '1d_vectors')
        mode_kwargs: Mode-specific parameters
        
    Returns:
        Dictionary with evaluation metrics
    """
    N = embeddings_0.shape[0]
    topk_metrics = topk_metrics or [1, 5, 10]
    max_topk = max(topk_metrics)

    results = {
        'mod0_to_mod1': [],
        'mod1_to_mod0': []
    }
    
    if mode == 'voting':
        # Pass embeddings_1 as a tensor for vectorized operations
        for i in tqdm(range(N)):
            query_emb = embeddings_0[i]
            result = retrieve_with_voting(query_emb, embeddings_1, i, mode_kwargs['num_votes'])
            results['mod0_to_mod1'].append(result)

    elif mode == '1d_vectors':
        results['mod0_to_mod1'] = retrieve_1d_vectors(embeddings_0, embeddings_1, max_topk=max_topk)
    
    return results


def compute_metrics(results: Dict, topk_metrics: List[int] = None) -> Dict:
    """
    Compute aggregate metrics from retrieval results.
    
    Args:
        results: Dictionary with retrieval results for each direction.
                 For voting mode: direction_results is a list of dicts
                 For 1d_vectors mode: direction_results is a single dict with tensors
        
    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    topk_metrics = sorted(set(topk_metrics or [1, 5, 10]))
    
    for direction, direction_results in results.items():
        if not direction_results:
            continue
        
        # Check if this is voting mode (list) or 1d_vectors mode (dict)
        if isinstance(direction_results, list):
            # Voting mode - existing logic
            N = len(direction_results)
            
            # Accuracy
            correct = sum(r['is_correct'] for r in direction_results)
            accuracy = 100.0 * correct / N
            
            # Average percentage of votes for predicted choice
            avg_predicted_vote_pct = np.mean([100.0 * r['predicted_votes'] / r['total_votes'] 
                                              for r in direction_results])
            
            # Average percentage of votes for correct choice
            avg_correct_vote_pct = np.mean([100.0 * r['correct_votes'] / r['total_votes'] 
                                            for r in direction_results])
            
            # Top-k accuracy
            topk_correct = {k: 0 for k in topk_metrics if k > 1}
            for r in direction_results:
                vote_dist = r['vote_distribution']
                top_choices = sorted(vote_dist.items(), key=lambda x: x[1], reverse=True)
                for k in topk_correct:
                    top_indices = [idx for idx, _ in top_choices[:k]]
                    if r['correct_idx'] in top_indices:
                        topk_correct[k] += 1
            
            metrics[direction] = {
                'accuracy': accuracy,
                'avg_predicted_vote_percentage': avg_predicted_vote_pct,
                'avg_correct_vote_percentage': avg_correct_vote_pct,
                'num_examples': N
            }
            for k, count in topk_correct.items():
                metrics[direction][f'top{k}_accuracy'] = 100.0 * count / N
            
        elif isinstance(direction_results, dict):
            # 1d_vectors mode
            predicted_idx = direction_results['predicted_idx']
            topk_indices = direction_results.get('topk_indices')
            
            # Convert to numpy if tensor
            if torch.is_tensor(predicted_idx):
                predicted_idx = predicted_idx.cpu().numpy()
            if torch.is_tensor(topk_indices):
                topk_indices = topk_indices.cpu().numpy()
            
            N = len(predicted_idx)
            correct_idx = np.arange(N)
            
            # Accuracy (top-1)
            correct = np.sum(predicted_idx == correct_idx)
            accuracy = 100.0 * correct / N
            
            # For 1d_vectors, we don't have vote percentages
            avg_predicted_vote_pct = None
            avg_correct_vote_pct = None
            
            # Compute top-k accuracies if we have indices
            topk_accuracies = {}
            
            if topk_indices is not None:
                for k in topk_metrics:
                    if k <= 1:
                        continue
                    k_slice = topk_indices[:, :min(k, topk_indices.shape[1])]
                    correct_in_topk = np.any(k_slice == correct_idx[:, np.newaxis], axis=1)
                    topk_accuracies[k] = 100.0 * np.sum(correct_in_topk) / N
            
            metrics[direction] = {
                'accuracy': accuracy,
                'avg_predicted_vote_percentage': avg_predicted_vote_pct,
                'avg_correct_vote_percentage': avg_correct_vote_pct,
                'num_examples': N
            }
            for k, value in topk_accuracies.items():
                metrics[direction][f'top{k}_accuracy'] = value
    
    return metrics


def print_metrics(metrics: Dict, topk_metrics: List[int] = None):
    """Print metrics in a readable format."""
    print("\n" + "="*70)
    print("CROSS-MODAL RETRIEVAL EVALUATION RESULTS")
    print("="*70)
    
    topk_metrics = sorted(set(topk_metrics or [1, 5, 10]))
    
    for direction, direction_metrics in metrics.items():
        print(f"\n{direction.upper().replace('_', ' ')}:")
        print(f"  Number of examples: {direction_metrics['num_examples']}")
        print(f"  Accuracy (Top-1): {direction_metrics['accuracy']:.2f}%")
        
        # Handle None values for metrics not available in all modes
        for k in topk_metrics:
            if k <= 1:
                continue
            key = f'top{k}_accuracy'
            value = direction_metrics.get(key)
            if value is not None:
                print(f"  Top-{k} Accuracy: {value:.2f}%")
            else:
                print(f"  Top-{k} Accuracy: N/A")
            
        if direction_metrics['avg_predicted_vote_percentage'] is not None:
            print(f"  Avg. vote % for predicted choice: {direction_metrics['avg_predicted_vote_percentage']:.2f}%")
        else:
            print(f"  Avg. vote % for predicted choice: N/A")
            
        if direction_metrics['avg_correct_vote_percentage'] is not None:
            print(f"  Avg. vote % for correct choice: {direction_metrics['avg_correct_vote_percentage']:.2f}%")
        else:
            print(f"  Avg. vote % for correct choice: N/A")
    
    print("\n" + "="*70)


def save_results(results: Dict, metrics: Dict, output_dir: str):
    """Save detailed results and metrics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    import json
    metrics_file = os.path.join(output_dir, 'retrieval_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'retrieval_results.json')
    # Convert to serializable format
    serializable_results = {}
    for direction, direction_results in results.items():
        if isinstance(direction_results, list):
            # Voting mode - list of dictionaries
            serializable_results[direction] = [
                {k: int(v) if isinstance(v, (np.integer, torch.Tensor)) else v 
                 for k, v in r.items()}
                for r in direction_results
            ]
        elif isinstance(direction_results, dict):
            # 1d_vectors mode - single dictionary with tensors
            serializable_results[direction] = {}
            for k, v in direction_results.items():
                if torch.is_tensor(v):
                    serializable_results[direction][k] = v.cpu().numpy().tolist()
                elif isinstance(v, np.ndarray):
                    serializable_results[direction][k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    serializable_results[direction][k] = int(v) if isinstance(v, np.integer) else float(v)
                else:
                    serializable_results[direction][k] = v
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved detailed results to: {results_file}")


@hydra.main(config_path="../../configs", config_name="evaluate_retrieval", version_base="1.3")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # Load environment variables
    load_dotenv()
    # Set random seed
    torch.manual_seed(cfg.eval.seed)
    np.random.seed(cfg.eval.seed)
    
    # Setup device
    device = cfg.runtime.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = to_absolute_path(cfg.model.checkpoint) if cfg.model.checkpoint is not None else None
    print(f"\nLoading model from: {checkpoint_path}")

    if cfg.model.get("no_train_agg", None) is not None:
        backbone_path = cfg.model.no_train_agg.backbone_chkp_path if cfg.model.no_train_agg.get("backbone_chkp_path", None) is not None else checkpoint_path
        model = create_aug_global_descriptor_model(
            backbone_chkp_path=backbone_path,
            descriptor_type=cfg.model.no_train_agg.descriptor_type,
            n_augs=cfg.model.no_train_agg.n_augs,
            device=device
        )

    elif cfg.model.checkpoint == "salad_og":
        model = create_salad_og_model()
    
    elif cfg.retrieval.mode == "1d_vectors":
        model = load_global_descriptor_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
        )

    elif cfg.retrieval.mode == "voting":
        model = ContrastiveLearningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=device,
            load_dino_weights=False
        )

    else:
        raise ValueError(f"Unsupported retrieval mode: {cfg.retrieval.mode}")

    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Create dataset
    print("\nCreating validation dataset...")
    if "dinov2" in cfg.model.checkpoint:
        cfg.data.image_size = (224, 224)
        
    val_dataset = create_datasets(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Dataset created with {len(val_dataset)} total samples")
    
    # Compute embeddings
    embeddings_0, embeddings_1, queries_and_database = compute_all_embeddings(
        model, val_loader, cfg.eval.num_examples, device
    )
    
    topk_metrics = cfg.retrieval.get("topk_metrics", [1, 5, 10])

    # Evaluate retrieval
    print(f"\nEvaluating retrieval with {cfg.retrieval.mode_kwargs.num_votes} votes per query...")
    results = evaluate_retrieval(
        embeddings_0, embeddings_1, 
        mode=cfg.retrieval.mode,
        mode_kwargs=cfg.retrieval.mode_kwargs,
        topk_metrics=topk_metrics
    )
    
    # Compute metrics
    metrics = compute_metrics(results, topk_metrics=topk_metrics)
    
    # Print results
    print_metrics(metrics, topk_metrics=topk_metrics)
    
    # Resolve output directory to absolute path
    output_dir = to_absolute_path(cfg.logging.output_dir)
    
    # Save results
    save_results(results, metrics, output_dir)
    
    # Save visualizations if requested
    num_visualizations = cfg.retrieval.get('num_visualizations', 0)
    if num_visualizations > 0:
        save_retrieval_visualizations(
            results=results,
            queries_and_database = queries_and_database,
            num_visualizations=num_visualizations,
            output_dir=output_dir,
            device=device
        )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

