import os 
import torch

from .my_attention import (
    CrossAttnLossAttnProcessor2_0, 
)

from .loss import (
    attn_maps, 
    CrossAttnHeatmapLoss, 
)

from .modules import (
    ControlNetModelForward, 
    Transformer2DModelForward,
    BasicTransformerBlockForward,
)


from diffusers.models import Transformer2DModel
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.controlnet import ControlNetModel




def setup_attn_processor(model, processor, **kwargs):
    print("--> set up attn processor")

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor: torch.nn.Module):
        if hasattr(module, "set_processor") and name.split('.')[-1].startswith('attn2'):
            print("name: {}".format(name))
            cur_processor = processor(**kwargs) 
            module.set_processor(cur_processor, _remove_lora=False)

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor)


def hook_fn(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, prefix="unet"):
    for name, module in model.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue

        if isinstance(module.processor, CrossAttnLossAttnProcessor2_0):
            module.processor.store_attn_map = True

        # hook = module.register_forward_hook(hook_fn(prefix + "." + name))
    
    return model


def set_layer_with_name_and_path(model, target_name="attn2", current_path=""):
    # if model.__class__.__name__ == 'UNet2DConditionModel':
    #     model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    if model.__class__.__name__ == 'ControlNetModel':
        model.forward = ControlNetModelForward.__get__(model, ControlNetModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        new_path = current_path + '.' + name if current_path else name
        set_layer_with_name_and_path(layer, target_name, new_path)
    
    return model
