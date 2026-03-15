from __future__ import annotations

from safetensors.torch import load_file
import torch
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
import MinkowskiEngine as ME
from tqdm import tqdm

from data.build import build_dataloader
from model.build import build_model
from .build import EVALUATION_REGISTRY
from evaluator import eval_utils
from common import misc
from common.load_utils import load_yaml
from util import scannet as scannet_utils

import os.path as osp
import itertools
import numpy as np

import csv

@EVALUATION_REGISTRY.register()
class SceneRetrieval():
    def __init__(self, cfg) -> None:
        super().__init__()
        
        set_seed(cfg.rng_seed)
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        
        task_config = cfg.task.get(cfg.task.name)
        
        self.ckpt_path = Path(task_config.ckpt_path)
        self.is_mmfe = 'mmfe' in str(self.ckpt_path).lower()
        
        self.output_csv = Path(task_config.get('output_csv', 'outputs/evaluation_results.csv'))
        
        self.modalities = list(task_config.scene_modalities)
        if not self.is_mmfe:
            self.modalities.append('object')
        
        modality_combinations = list(itertools.combinations(self.modalities, 2))
        same_element_combinations = [(item, item) for item in self.modalities]
        modality_combinations.extend(same_element_combinations)
        self.modality_combinations = modality_combinations
                
        key = "val"
        self.dataset_name = misc.rgetattr(task_config, key)[0]
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]
        
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        
        if self.is_mmfe:
            self.model = self._build_mmfe_model(cfg, task_config)
        else:
            self.model = build_model(cfg)
        
        self.data_loader = build_dataloader(cfg, split=key, is_training=False)
        
        self.pcl_sparse_source = task_config.get('pcl_sparse_source', 'coordinates')
        if self.pcl_sparse_source == 'mesh':
            import albumentations as A
            data_cfg = cfg.data[self.dataset_name]
            self._mesh_base_dir = data_cfg.base_dir
            self._mesh_voxel_size = data_cfg.mesh_voxel_size
            color_mean_std = load_yaml(osp.join(data_cfg.process_dir, 'color_mean_std.yaml'))
            self._mesh_normalize_color = A.Normalize(
                mean=tuple(color_mean_std["mean"]),
                std=tuple(color_mean_std["std"]),
            )
        
        self.model, self.data_loader = self.accelerator.prepare(self.model, self.data_loader)
        
        if not self.is_mmfe:
            self.load_from_ckpt()
    
    def _build_mmfe_model(self, cfg, task_config):
        import sys
        mmfe_src = osp.abspath(osp.join(osp.dirname(__file__), '..', 'third_party', 'mmfe', 'src'))
        if mmfe_src not in sys.path:
            sys.path.insert(0, mmfe_src)

        from global_descriptors.global_descriptor_models import load_global_descriptor_model_from_checkpoint
        from global_descriptors.global_descriptor_modules import GlobalDescriptorLightningModule

        self.logger.info(f"Loading MMFE model from {self.ckpt_path}")
        inner_model = load_global_descriptor_model_from_checkpoint(str(self.ckpt_path))
        lightning_module = GlobalDescriptorLightningModule(model=inner_model)

        base_dir = cfg.data[self.dataset_name].base_dir
        mmfe_config = task_config.get('mmfe', {})
        image_size = list(mmfe_config.get('image_size', [224, 224]))

        cc = inner_model.creation_config
        backbone_name = cc["backbone"]["name"] if cc and "backbone" in cc else ""
        if "dinov2" in backbone_name:
            image_size = [s // 14 * 14 for s in image_size]
            self.logger.info(
                f"DINOv2 backbone detected — image_size adjusted to {image_size} "
                f"(nearest multiple of 14)"
            )

        lightning_module.setup_crossover(
            base_dir=base_dir,
            image_size=tuple(image_size),
            floorplan_img_name=mmfe_config.get('floorplan_img_name', 'mmfe_floorplan.png'),
            point_source=mmfe_config.get('point_source', 'density'),
            density_name=mmfe_config.get('density_name', 'density.png'),
            scene_modalities=list(task_config.scene_modalities),
            debug_dir=mmfe_config.get('debug_dir', None),
            point_size=mmfe_config.get('point_size', 1),
        )

        self.logger.info("MMFE model configured for CrossOver evaluation")
        return lightning_module

    def _build_pcl_sparse_from_mesh(self, data_dict):
        """Build an ME.SparseTensor by loading mesh PLY files per scan."""
        scan_ids = data_dict['scan_id']
        all_coords = []
        all_feats = []

        for scan_id in scan_ids:
            mesh_file = osp.join(self._mesh_base_dir, scan_id, 'floor+obj.ply')
            # mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
            mesh_vertices = scannet_utils.read_mesh_vertices_rgb_mmfe(mesh_file, num_samples=len(data_dict['coordinates']))

            points = mesh_vertices[:, 0:3]
            # Undo the -90° Z rotation baked into floor+obj.ply by makeShapeAndLayoutMesh
            points = np.column_stack([-points[:, 1], points[:, 0], points[:, 2]])
            feats = mesh_vertices[:, 3:]
            pseudo_image = feats.astype(np.uint8)[np.newaxis, :, :]
            feats = np.squeeze(
                self._mesh_normalize_color(image=pseudo_image)["image"]
            )

            _, sel = ME.utils.sparse_quantize(
                points / self._mesh_voxel_size, return_index=True,
            )
            coords = np.floor(points[sel] / self._mesh_voxel_size)
            coords -= coords.min(0)
            feats = feats[sel]

            all_coords.append(coords)
            all_feats.append(feats)

        coords, feats = ME.utils.sparse_collate(all_coords, all_feats)
        return ME.SparseTensor(
            coordinates=coords,
            features=feats.to(torch.float32),
            device=self.accelerator.device,
        )

    def forward(self, data_dict):
        if self.is_mmfe:
            return self.model.crossover_forward(data_dict)
        return self.model(data_dict)
    
    @torch.no_grad()
    def inference_step(self):
        self.model.eval()
        
        loader = self.data_loader
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        self.logger.info('Running validation...')
        
        outputs = []
        for iter, data_dict in enumerate(loader):
            if self.pcl_sparse_source == 'mesh':
                data_dict['pcl_sparse'] = self._build_pcl_sparse_from_mesh(data_dict)
            else:
                data_dict['pcl_sparse'] = ME.SparseTensor(
                        coordinates=data_dict['coordinates'],
                        features=data_dict['features'].to(torch.float32),
                        device=self.accelerator.device)
            
            data_dict = self.forward(data_dict)
            
            num_scans = len(data_dict['scan_id'])
            
            for idx in range(num_scans):
                output = { 'scan_id' : data_dict['scan_id'][idx], 'scene_label': data_dict['scene_label'][idx], 'outputs' : {}, 'masks' : {}}
                for modality in self.modalities:                                
                    output['outputs'][modality] = data_dict['embeddings'][modality][idx]
                    output['masks'][modality] = data_dict['scene_masks'][modality][idx]      
                outputs.append(output)             
            pbar.update(1)

        return outputs
 
    def eval(self, output_dict):
        scan_data = np.array([{ 'scan_id': output_data['scan_id'], 'label' : output_data['scene_label']} for output_data in output_dict])
        unique_labels = {data['label'] for data in scan_data}

        # --- NEW: Initialize a list to hold our rows for the CSV ---
        csv_data = []

        for src_modality, ref_modality in self.modality_combinations:
            if src_modality == 'object' or ref_modality == 'object':
                continue
            
            src_embed = torch.stack([output_data['outputs'][src_modality] for output_data in output_dict])
            ref_embed = torch.stack([output_data['outputs'][ref_modality] for output_data in output_dict])
        
            src_mask  = torch.stack([output_data['masks'][src_modality] for output_data in output_dict])
            ref_mask  = torch.stack([output_data['masks'][ref_modality] for output_data in output_dict])
            mask = torch.logical_and(src_mask, ref_mask).reshape(-1,)
            
            if mask.sum() == 0:
                continue
            
            src_embed = src_embed[mask]
            ref_embed = ref_embed[mask]
            
            scan_data_masked = scan_data[mask.cpu().numpy()]
            
            sim = torch.softmax(src_embed @ ref_embed.t(), dim=-1)
            rank_list = torch.argsort(1.0 - sim, dim = 1)
            top_k_indices = rank_list[:, :20]
                
            correct_index = torch.arange(src_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
            matches = top_k_indices == correct_index   
            
            # --- 1. Recall ---
            recall_top1 = matches[:, 0].float().mean() * 100.
            recall_top5 = matches[:, :5].any(dim=1).float().mean() * 100.
            recall_top10 = matches[:, :10].any(dim=1).float().mean() * 100.
            recall_top20 = matches.any(dim=1).float().mean() * 100.
            
            message = f"{src_modality} -> {ref_modality}:" 
            self.logger.info(message)
            
            message = f'Recall: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}, top20 - {recall_top20:.2f}'
            self.logger.info(message)
            
            # Append Recall to CSV data
            pair_name = f"{src_modality} -> {ref_modality}"
            csv_data.append([pair_name, 'Recall', f"{recall_top1:.2f}", "", f"{recall_top5:.2f}", f"{recall_top10:.2f}", f"{recall_top20:.2f}"])
            
            # --- 2. Temporal eval ---
            recall_top1, recall_top5, recall_top10 = eval_utils.evaluate_temporal_scene_matching(rank_list.cpu().numpy().tolist(), scan_data_masked, self.data_loader.dataset.get_temporal_scan_pairs())
            message = f'Temporal: top1 - {recall_top1:.2f}, top5 - {recall_top5:.2f}, top10 - {recall_top10:.2f}'
            self.logger.info(message)  
            
            # Append Temporal to CSV data
            csv_data.append(["", 'Temporal', f"{recall_top1:.2f}", "", f"{recall_top5:.2f}", f"{recall_top10:.2f}", ""])
            
            if self.dataset_name == 'Scannet':
                # --- 3. Category eval ---
                st_recall_top1, st_recall_top5, st_recall_top10 = eval_utils.calculate_scene_label_recall(rank_list.cpu().numpy().tolist(), scan_data_masked)
                st_recall_top1 *= 100.
                st_recall_top5 *= 100.
                st_recall_top10 *= 100.
                
                message  =  f"Category: top1 - {st_recall_top1:.2f}, top5 - {st_recall_top5:.2f}, top10 - {st_recall_top10:.2f}"
                self.logger.info(message)     
                
                # Append Category to CSV data
                csv_data.append(["", 'Category', f"{st_recall_top1:.2f}", "", f"{st_recall_top5:.2f}", f"{st_recall_top10:.2f}", ""])
                
                # --- 4. Intra-Category eval ---
                ic_recall_top1, ic_recall_top3, ic_recall_top5 = eval_utils.evaluate_intra_category_scene_matching(scan_data_masked, src_embed, ref_embed, unique_labels)
                message  =  f"Intra-Category: top1 - {ic_recall_top1:.2f}, top3 - {ic_recall_top3:.2f}, top5 - {ic_recall_top5:.2f}"
                self.logger.info(message)  
                
                # Append Intra-Category to CSV data
                csv_data.append(["", 'Intra-Category', f"{ic_recall_top1:.2f}", f"{ic_recall_top3:.2f}", f"{ic_recall_top5:.2f}", "", ""])

        # --- NEW: Write the collected data to a CSV file ---
        # You can customize this filename to include timestamps or model names if needed
        csv_filename = self.output_csv
        csv_filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Source -> Target', 'Metric Type', 'Top 1', 'Top 3', 'Top 5', 'Top 10', 'Top 20'])
            # Write all the data rows
            writer.writerows(csv_data)
            
        self.logger.info(f"Successfully saved structured results to {csv_filename}")
                   
    def load_from_ckpt(self):
        if self.ckpt_path.exists():
            self.logger.info(f"Loading from {self.ckpt_path}")
            # Load model weights from safetensors files
            ckpt = osp.join(self.ckpt_path, 'model.safetensors')
            ckpt = load_file(ckpt,  device = str(self.accelerator.device))
            self.model.load_state_dict(ckpt)
            self.logger.info(f"Successfully loaded from {self.ckpt_path}")
        
        else:
            raise FileNotFoundError
    
    def run(self):
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Inference Step -- CrossOver
        output_dict = self.inference_step()
        self.logger.info('Scene Retrieval Evaluation (Unified Scene CrossOver)...')
        self.eval(output_dict)