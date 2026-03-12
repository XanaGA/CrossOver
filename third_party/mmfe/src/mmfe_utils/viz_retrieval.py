"""
Visualization utilities for retrieval evaluation.

This module provides functions to visualize cross-modal retrieval results,
showing query images alongside their top-k retrieved matches.
"""

import os
from typing import List, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloading.unified_dataset import UnifiedDataset


def denormalize_image(img, mean: List[float] = [0.485, 0.456, 0.406], 
                      std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Convert an image to numpy array in [0, 1] range.
    Handles PIL Images, numpy arrays, and tensors (normalized or not).
    
    Args:
        img: Image as PIL Image, numpy array, or torch.Tensor
        mean: Mean values for normalization (if image is normalized tensor)
        std: Std values for normalization (if image is normalized tensor)
        
    Returns:
        Image as numpy array in [0, 1] range, shape (H, W, C)
    """
    # Handle PIL Image
    if isinstance(img, Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0
        if img_np.ndim == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)
        return img_np
    
    # Handle numpy array
    if isinstance(img, np.ndarray):
        img_np = img.astype(np.float32)
        # If in [0, 255] range, normalize to [0, 1]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        # Ensure 3 channels
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        # Ensure (H, W, C) format
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        return np.clip(img_np, 0, 1)
    
    # Handle torch.Tensor
    if isinstance(img, torch.Tensor):
        img_tensor = img.clone()
        
        # Check if normalized (values typically in [-2, 2] range after normalization)
        is_normalized = img_tensor.min() < -0.5 or img_tensor.max() > 1.5
        
        if is_normalized:
            # Denormalize
            if img_tensor.ndim == 3:
                if img_tensor.shape[0] == 3:  # (C, H, W)
                    img_tensor = img_tensor.permute(1, 2, 0)  # (H, W, C)
            
            mean_tensor = torch.tensor(mean).view(1, 1, 3).to(img_tensor.device)
            std_tensor = torch.tensor(std).view(1, 1, 3).to(img_tensor.device)
            
            img_denorm = img_tensor * std_tensor + mean_tensor
            img_denorm = torch.clamp(img_denorm, 0, 1)
        else:
            # Already in [0, 1] range, just ensure correct format
            if img_tensor.ndim == 3:
                if img_tensor.shape[0] == 3:  # (C, H, W)
                    img_tensor = img_tensor.permute(1, 2, 0)  # (H, W, C)
            img_denorm = torch.clamp(img_tensor, 0, 1)
        
        return img_denorm.cpu().numpy()
    
    raise TypeError(f"Unsupported image type: {type(img)}")


def get_topk_indices_for_query(query_idx: int, results: Dict, direction: str, topk: int = 10) -> List[int]:
    """
    Get top-k retrieved indices for a query.
    
    Args:
        query_idx: Index of the query
        results: Results dictionary from evaluate_retrieval
        direction: Direction of retrieval ('mod0_to_mod1' or 'mod1_to_mod0')
        topk: Number of top results to return
        
    Returns:
        List of top-k indices
    """
    direction_results = results.get(direction, [])
    
    if isinstance(direction_results, list):
        # Voting mode
        if query_idx >= len(direction_results):
            return []
        result = direction_results[query_idx]
        vote_dist = result.get('vote_distribution', {})
        # Sort by vote count and get top-k
        sorted_votes = sorted(vote_dist.items(), key=lambda x: x[1], reverse=True)
        topk_indices = [idx for idx, _ in sorted_votes[:topk]]
        return topk_indices
    elif isinstance(direction_results, dict):
        # 1d_vectors mode
        topk_indices_tensor = direction_results.get('topk_indices')
        if topk_indices_tensor is None:
            return []
        
        # Convert to numpy if tensor
        if torch.is_tensor(topk_indices_tensor):
            topk_indices_tensor = topk_indices_tensor.cpu().numpy()
        
        if query_idx >= len(topk_indices_tensor):
            return []
        
        # Get top-k for this query
        topk_slice = topk_indices_tensor[query_idx, :min(topk, topk_indices_tensor.shape[1])]
        return topk_slice.tolist()
    
    return []


def visualize_retrieval_results(query_img, ground_truth_img, retrieved_imgs: List,
                                retrieved_indices: List[int], correct_idx: int, output_path: str):
    """
    Create a visualization showing the query image and top-k retrieved images.
    
    Args:
        query_img: Query image (PIL Image, numpy array, or torch.Tensor)
        ground_truth_img: Ground truth image (PIL Image, numpy array, or torch.Tensor)
        query_img: Query image (PIL Image, numpy array, or torch.Tensor)
        retrieved_imgs: List of retrieved images (PIL Images, numpy arrays, or torch.Tensors)
        retrieved_indices: List of indices of retrieved images
        correct_idx: Correct index (to highlight if present)
        output_path: Path to save the visualization
    """
    # Denormalize query image
    query_np = denormalize_image(query_img)
    ground_truth_np = denormalize_image(ground_truth_img)

    # Denormalize retrieved images
    retrieved_np_list = [denormalize_image(img) for img in retrieved_imgs]
    
    # Create figure with query on left, retrieved images in a grid on right
    num_retrieved = len(retrieved_np_list)
    fig = plt.figure(figsize=(20, 4))
    
    # Plot query image
    ax = plt.subplot(1, num_retrieved + 2, 1)
    ax.imshow(query_np)
    ax.set_title(f'Query', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot ground truth image
    ax = plt.subplot(1, num_retrieved + 2, 2)
    ax.imshow(ground_truth_np)
    ax.set_title(f'Ground Truth', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Plot retrieved images
    for i, (ret_img, ret_idx) in enumerate(zip(retrieved_np_list, retrieved_indices)):
        ax = plt.subplot(1, num_retrieved + 2, i + 3)
        ax.imshow(ret_img)
        
        # Highlight correct match
        if ret_idx == correct_idx:
            title = f'Retrieved #{i+1}\n✓ CORRECT'
            ax.set_title(title, fontsize=10, fontweight='bold', color='green')
            # Add green border
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            title = f'Retrieved #{i+1}'
            ax.set_title(title, fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_retrieval_visualizations(results: Dict, queries_and_database: List[Dict],
                                 num_visualizations: int, output_dir: str, device: str):
    """
    Save visualization images for retrieval queries.
    
    Args:
        results: Results dictionary from evaluate_retrieval
        queries_and_database: List of query and database images
        num_visualizations: Number of visualizations to create
        output_dir: Output directory for saving visualizations
        device: Device to use
    """
    if num_visualizations <= 0:
        return
    
    viz_dir = os.path.join(output_dir, 'retrieval_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nCreating {num_visualizations} retrieval visualizations...")
    
    # Get top-k indices for each mode
    direction = 'mod0_to_mod1'  # Visualize modality 0 -> modality 1 retrieval
    direction_results = results.get(direction, [])
    
    if not direction_results:
        print(f"Warning: No results found for direction {direction}, skipping visualizations")
        return
    
    # Determine number of queries available
    if isinstance(direction_results, list):
        num_queries = len(direction_results)
    elif isinstance(direction_results, dict):
        predicted_idx = direction_results.get('predicted_idx')
        if torch.is_tensor(predicted_idx):
            num_queries = len(predicted_idx)
        elif isinstance(predicted_idx, np.ndarray):
            num_queries = len(predicted_idx)
        else:
            num_queries = 0
    else:
        num_queries = 0
    
    if num_queries == 0:
        print("Warning: No queries available for visualization")
        return
    
    # Limit number of visualizations to available queries
    num_visualizations = min(num_visualizations, num_queries)
    
    # Collect images from dataset
    # We need to get the original images (before normalization) from the dataset
    query_indices = list(range(num_visualizations))
    
    for query_idx in tqdm(query_indices, desc="Creating visualizations"):
        try:
            # Get query image from dataset (original, before normalization)
            query_item = queries_and_database[query_idx]
            query_img= query_item['modality_0']
            
            # Get top-10 retrieved indices
            top10_indices = get_topk_indices_for_query(query_idx, results, direction, topk=10)
            
            if not top10_indices:
                print(f"Warning: No retrieved indices for query {query_idx}, skipping")
                continue
            
            # Get retrieved images from dataset
            retrieved_imgs = []
            for ret_idx in top10_indices:
                try:
                    ret_item = queries_and_database[ret_idx]
                    ret_img_original = ret_item.get('original_modality_1')
                    if ret_img_original is None:
                        # Fallback to modality_1 if original not available
                        ret_img_original = ret_item['modality_1']
                    retrieved_imgs.append(ret_img_original)
                except Exception as e:
                    print(f"Warning: Could not load retrieved image {ret_idx}: {e}")
                    # Use a placeholder black image
                    # Try to infer shape from query image or first retrieved image
                    if len(retrieved_imgs) > 0:
                        ref_img = retrieved_imgs[0]
                    else:
                        ref_img = query_img
                    
                    # Create placeholder based on type
                    if isinstance(ref_img, torch.Tensor):
                        placeholder = torch.zeros_like(ref_img)
                    elif isinstance(ref_img, np.ndarray):
                        placeholder = np.zeros_like(ref_img)
                    elif isinstance(ref_img, Image.Image):
                        placeholder = Image.new('RGB', ref_img.size, color='black')
                    else:
                        # Default: create a black PIL image
                        placeholder = Image.new('RGB', (256, 256), color='black')
                    retrieved_imgs.append(placeholder)
            
            # Get correct index
            if isinstance(direction_results, list):
                correct_idx = direction_results[query_idx].get('correct_idx', query_idx)
            else:
                correct_idx = query_idx

            ground_truth_img = queries_and_database[correct_idx].get('original_modality_1')
            
            # Create visualization
            output_path = os.path.join(viz_dir, f'query_{query_idx:04d}.png')
            visualize_retrieval_results(
                query_img=query_img,
                ground_truth_img=ground_truth_img,
                retrieved_imgs=retrieved_imgs,
                retrieved_indices=top10_indices,
                correct_idx=correct_idx,
                output_path=output_path
            )
            
        except Exception as e:
            print(f"Error creating visualization for query {query_idx}: {e}")
            continue
    
    print(f"\nSaved {len([f for f in os.listdir(viz_dir) if f.endswith('.png')])} visualizations to: {viz_dir}")

