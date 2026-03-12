from typing import Optional
import torch
import torch.nn as nn

from mmfe_utils.dino_utils import load_dino

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

DINOV3_ARCHS = {
    'dinov3_vits16': 384,
    'dinov3_vits16plus': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vith16plus': 1024,
    'dinov3_vit7b16': 1024,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False,
            loading_config: dict = {}
        ):
        super().__init__()

        # assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model_name = model_name
        if model_name.startswith("dinov2"):
            local_path = loading_config.get("local_path", None)
            self.model = torch.hub.load('facebookresearch/dinov2', model_name) if local_path is None else torch.hub.load(local_path, model_name, source="local", pretrained=False)
            self.num_channels = DINOV2_ARCHS[model_name]
        elif model_name.startswith("dinov3"):
            self.model = load_dino(model_name, **loading_config)
            self.num_channels = DINOV3_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        if self.model_name.startswith("dinov3"):
            self._freeze_blocks()

    def _freeze_blocks(self):
        total = len(self.model.blocks)

        # Freeze early blocks
        for blk in self.model.blocks[:total - self.num_trainable_blocks]:
            for p in blk.parameters():
                p.requires_grad = False

        # Unfreeze last blocks
        for blk in self.model.blocks[total - self.num_trainable_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        if self.model_name.startswith("dinov3"):
            # x = self.model.forward_features(x)

            x, hw_tuple = self.model.prepare_tokens_with_masks(x, None)
            rope_sincos = self.model.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
            for _, blk in enumerate(self.model.blocks[:-self.num_trainable_blocks]):
                x = blk(x, rope_sincos)
            x = x.detach()
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                x = blk(x, rope_sincos)

            if self.norm_layer:
                x = self.model.norm(x)

            t = x[:, 0]
            f = x[:, self.model.n_storage_tokens + 1 :]

            f = f.reshape((B, H // 16, W // 16, self.num_channels)).permute(0, 3, 1, 2)
        else:
            x = self.model.prepare_tokens_with_masks(x)
        
            # First blocks are frozen
            with torch.no_grad():
                for blk in self.model.blocks[:-self.num_trainable_blocks]:
                    x = blk(x)
            x = x.detach()

            # Last blocks are trained
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                x = blk(x)

            if self.norm_layer:
                x = self.model.norm(x)

            t = x[:, 0]
            f = x[:, 1:]

            f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        

        if self.return_token:
            return f, t
        return f