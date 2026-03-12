def create_rotation_grid_old(
    img_0: torch.Tensor, 
    img_1: torch.Tensor, 
    features_0_rotated: torch.Tensor, 
    features_1_rotated: torch.Tensor,
    features_0_computed: torch.Tensor,
    features_1_computed: torch.Tensor,
) -> np.ndarray:
    """Create a 2x3 grid showing rotation effects on features."""
    
    # Convert tensors to numpy images
    img_0_np = tensor_to_numpy_image(img_0)
    img_1_np = tensor_to_numpy_image(img_1)
    
    # Get image dimensions
    img_h, img_w = img_0_np.shape[:2]
    
    # Create PCA viz for rotated features
    pca_0_rotated, pca_1_rotated, pca_0_computed, pca_1_computed = pca_rgb([features_0_rotated.cpu(), 
                                                                    features_1_rotated.cpu(),
                                                                    features_0_computed.cpu(),
                                                                    features_1_computed.cpu()])
    # pca_0_rotated = pca_rgb([features_0_rotated.cpu()])[0]
    # pca_1_rotated = pca_rgb([features_1_rotated.cpu()])[0]
    # pca_0_computed = pca_rgb([features_0_computed.cpu()])[0]
    # pca_1_computed = pca_rgb([features_1_computed.cpu()])[0]

    # Create difference images
    cmap_0_rotated = get_color_map(features_0_rotated.cpu(), features_0_computed.cpu())
    cmap_1_rotated = get_color_map(features_1_rotated.cpu(), features_1_computed.cpu())

    # Resize PCA viz to match image size
    pca_0_rotated = cv2.resize(pca_0_rotated, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    pca_1_rotated = cv2.resize(pca_1_rotated, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    pca_0_computed = cv2.resize(pca_0_computed, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    pca_1_computed = cv2.resize(pca_1_computed, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    cmap_0_rotated = cv2.resize(cmap_0_rotated, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    cmap_1_rotated = cv2.resize(cmap_1_rotated, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Create separators
    sep_h = np.ones((img_h, 10, 3), dtype=np.float32)
    sep_v = np.ones((10, img_w * 4 + 30, 3), dtype=np.float32)
    
    # Create rows
    # Row 1: Modality 0 - [original rotated, rotated features, computed features]
    row_1 = np.concatenate([img_0_np, sep_h, pca_0_rotated, sep_h, pca_0_computed, sep_h, cmap_0_rotated], axis=1)
    
    # Row 2: Modality 1 - [original rotated, rotated features, computed features] 
    row_2 = np.concatenate([img_1_np, sep_h, pca_1_rotated, sep_h, pca_1_computed, sep_h, cmap_1_rotated], axis=1)
    
    # Combine rows
    grid = np.concatenate([row_1, sep_v, row_2], axis=0)
    
    return grid

def create_rotation_video_old(
    model: ContrastiveLearningModule,
    dataset: UnifiedDataset,
    sample_idx: int,
    output_dir: str,
    num_rotations: int = 36,
    device: str = 'cuda'
) -> str:
    """Create a rotation video for a specific sample."""
    
    print(f"Processing sample {sample_idx}...")
    
    try:
        # Get sample from dataset
        sample = dataset[sample_idx]
        modality_0 = sample['modality_0'].unsqueeze(0).to(device)  # Add batch dimension
        modality_1 = sample['modality_1'].unsqueeze(0).to(device)
        
        # Get features for 0-degree rotation (reference)
        with torch.no_grad():
            features_0_ref, features_1_ref = model.forward(modality_0, modality_1)
            
        print(f"  Reference features shape: {features_0_ref.shape}, {features_1_ref.shape}")
        
    except Exception as e:
        print(f"  Error loading sample {sample_idx}: {e}")
        raise
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate rotation frames
        angles = np.linspace(0, 360, num_rotations, endpoint=False)

        for i, angle in enumerate(angles):
            print(f"  Processing rotation {i+1}/{num_rotations}: {angle:.1f}°")
            
            # Rotate input images
            img_0_rotated = rotate_tensor(modality_0[0], angle)
            img_1_rotated = rotate_tensor(modality_1[0], angle)
            
            # Compute features for rotated inputs
            with torch.no_grad():
                features_0_computed, features_1_computed = model.forward(
                    img_0_rotated.unsqueeze(0), 
                    img_1_rotated.unsqueeze(0)
                )
                features_0_computed = features_0_computed[0]
                features_1_computed = features_1_computed[0]

            
            # Handle different projection head types and prepare features for visualization
            if model.model.encoder.projection_head_type == "cnn" or "dino" in model.model.encoder.backbone_name:
                # For CNN heads, features are already 3D (C, H, W)
                features_0_rotated = rotate_tensor(features_0_ref[0], angle)
                features_1_rotated = rotate_tensor(features_1_ref[0], angle)
            else:
                raise ValueError(f"1D features non suported yet: {model.model.encoder.projection_head_type}")
                # For linear/MLP heads, we need to reshape to 2D for visualization
                feat_dim = features_0_computed.shape[0]
                feat_h = int(np.sqrt(feat_dim))
                feat_w = feat_dim // feat_h
                
                if feat_h * feat_w == feat_dim:
                    # Reshape computed features
                    features_0_computed = features_0_computed[:feat_h*feat_w].view(feat_h, feat_w).unsqueeze(0)
                    features_1_computed = features_1_computed[:feat_h*feat_w].view(feat_h, feat_w).unsqueeze(0)
                    
                    # Reshape and rotate reference features for comparison
                    features_0_ref_reshaped = features_0_ref[0][:feat_h*feat_w].view(feat_h, feat_w).unsqueeze(0)
                    features_1_ref_reshaped = features_1_ref[0][:feat_h*feat_w].view(feat_h, feat_w).unsqueeze(0)
                    
                    features_0_rotated = rotate_tensor(features_0_ref_reshaped, angle)
                    features_1_rotated = rotate_tensor(features_1_ref_reshaped, angle)
                else:
                    print(f"  Warning: Cannot reshape features of dimension {feat_dim} to square")
                    continue
            
            # Create grid visualization
            grid = create_rotation_grid(
                img_0_rotated, img_1_rotated,
                features_0_rotated, features_1_rotated,
                features_0_computed, features_1_computed,
            )
            
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            grid_to_save = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
            grid_to_save = cv2.cvtColor(grid_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, grid_to_save)
        
        # Create video from frames
        video_path = os.path.join(output_dir, f"rotation_sample_{sample_idx}.mp4")
        os.makedirs(output_dir, exist_ok=True)
        
        # Use OpenCV to create video
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
        if frame_files:
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
            h, w, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
            # Write frames
            for frame_file in frame_files:
                frame = cv2.imread(os.path.join(temp_dir, frame_file))
                video_writer.write(frame)
            
            video_writer.release()
            print(f"  Video saved: {video_path}")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return video_path