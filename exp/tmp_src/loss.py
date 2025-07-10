import os 
from diffusers.models.attention import Attention
from .my_attention import (
    CrossAttnLossAttnProcessor2_0, 
)

import torch
import torch.nn.functional as F

from einops import rearrange
from PIL import Image 
import numpy as np 
import logging 

logger = logging.getLogger(__name__)


attn_maps = {}

NUM_KEYPOINT = 17

def CrossAttnHeatmapLoss(
        controlnet, target, timestep, 
        token_placeholder_mask, 
        keypoint_prompt_vis_mask, 
        attn_size=64, 
        attn_target_type="whole",
        loss_type="bce", 
        loss_start_timestep=None, 
        loss_end_timestep=None, 
        select_layer_type=None, 
    ):
    
    # print("cross-attn heatmap loss")
    bz = target.shape[0]
    token_placeholder_mask = token_placeholder_mask.reshape((bz, 77, 1, 1))

    # when caption is "", cross-attntion loss is None
    if token_placeholder_mask.sum() == 0:
        return None
    
    ### prepare attention maps
    attn_probs = {}
    for name, module in controlnet.named_modules():
        if isinstance(module, Attention) and name.split('.')[-1].startswith('attn2'):
            cur_timestep = module.processor.timestep
            a = module.processor.attn_map.mean(dim=1)  # (B, Head, 77, H, W) -> (B, 77, H, W)
            # only select some layers cross-attention map 
            if select_layer_type and (not select_layer_type in name): 
                continue
            else:
                attn_probs[name] = F.interpolate(a, size=(attn_size, attn_size), mode="bilinear", align_corners=True)

    avg_attn_maps = torch.stack([v for k,v in attn_probs.items()], dim=0)
    avg_attn_maps = torch.mean(avg_attn_maps, dim=0)  # [batch, 77, H, W]

    # normalize 
    avg_attn_min = torch.min(avg_attn_maps.reshape(bz, -1), dim=1)[0].reshape((bz, 1, 1, 1))
    avg_attn_max = torch.max(avg_attn_maps.reshape(bz, -1), dim=1)[0].reshape((bz, 1, 1, 1))
    normed_avg_attn_maps = (avg_attn_maps - avg_attn_min) / (avg_attn_max - avg_attn_min).clamp(min=1e-6)
    
    located_avg_attn_maps = torch.masked_select(normed_avg_attn_maps, token_placeholder_mask.to(torch.bool))

    ### prepare target maps
    if attn_target_type == "whole":
        target = F.interpolate(target[:, :1, :, :], size=(attn_size, attn_size), mode="bilinear", align_corners=True)
        target_mask = torch.sum(token_placeholder_mask, dim=1, keepdim=True)
        located_target = torch.masked_select(target, mask=target_mask.to(torch.bool))
    elif attn_target_type == "partition":
        target = F.interpolate(target, size=(attn_size, attn_size), mode="bilinear", align_corners=True)
        assert torch.all(torch.sum(token_placeholder_mask, dim=1).squeeze() == torch.sum(keypoint_prompt_vis_mask, dim=1).squeeze())
        keypoint_prompt_vis_mask = keypoint_prompt_vis_mask.reshape(-1, 17, 1, 1)
        target_mask = keypoint_prompt_vis_mask
        located_target = torch.masked_select(target, mask=keypoint_prompt_vis_mask.to(torch.bool))
    else:
        raise NotImplementedError
    
    # weighted by timestep
    timestep_weight_map = None
    if not loss_start_timestep is None and not loss_end_timestep is None:
        timestep_weight_map = get_timestep_weight(target, target_mask, timestep, loss_start_timestep, loss_end_timestep)

    if loss_type == "bce":
        loss_function = F.binary_cross_entropy
    elif loss_type == "mse":
        loss_function = F.mse_loss
    else:
        raise NotImplementedError

    if not timestep_weight_map is None:
        loss_weight = timestep_weight_map
        attn_loss = loss_function(located_avg_attn_maps.clamp(min=0, max=1), located_target.clamp(min=0, max=1), reduction="none")
        attn_loss = torch.mean(loss_weight * attn_loss)
    else:
        attn_loss = loss_function(located_avg_attn_maps.clamp(min=0, max=1), located_target.clamp(min=0, max=1), reduction="mean", )

    return attn_loss


def get_timestep_weight(target, target_mask, timestep, loss_start_timestep, loss_end_timestep):
    """
    return the weight, computed by timestep [b, ]
    """
    device = timestep.device
    weight_map = torch.ones_like(target).to(device)
        
    t_weight = torch.where(torch.logical_and(timestep>loss_start_timestep, timestep<loss_end_timestep), 1, 0)
    weight_map = weight_map * t_weight.reshape(-1, 1, 1, 1)
    weight_map = torch.masked_select(weight_map, mask=target_mask.to(torch.bool))
    return weight_map
