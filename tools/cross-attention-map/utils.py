import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers.models import Transformer2DModel
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
)

from modules import (
    attn_maps, 
    attn_call, 
    attn_call2_0, 
    UNet2DConditionModelForward, 
    Transformer2DModelForward, 
    BasicTransformerBlockForward,  
    ControlNetModelForward, 
)


def cross_attn_init():
    ########## attn_call is faster ##########
    AttnProcessor.__call__ = attn_call  # not used
    AttnProcessor2_0.__call__ = attn_call2_0


def hook_fn(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet, prefix="unet"):
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(prefix + "." + name))
    
    return unet


def set_layer_with_name_and_path(model, target_name="attn2", current_path=""):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

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


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        try:
            token = tokenizer.decoder[text_input_id.item()]
        except:
            token = tokenizer.decode(text_input_id.item())
        tokens.append(token)
    return tokens


def resize_and_save(tokenizer, prompt, timestep=None, path=None, max_height=256, max_width=256, save_path='attn_maps', interpolation="bilinear"):
    resized_map = None

    if interpolation == "bilinear":
        interpolate_args = {"mode": "bilinear", "align_corners": False, }
    elif interpolation == "nearest": 
        interpolate_args = {"mode": "nearest"}
    else:
        raise NotImplementedError  

    if path is None:
        for path_ in list(attn_maps[timestep].keys()):
            
            value = attn_maps[timestep][path_]
            value = torch.mean(value,axis=0).squeeze(0)
            seq_len, h, w = value.shape
            max_height = max(h, max_height)
            max_width = max(w, max_width)
            value = F.interpolate(
                value.to(dtype=torch.float32).unsqueeze(0),
                size=(max_height, max_width),
                mode='nearest',  
                # mode="bilinear", 
                # align_corners=True, 
            ).squeeze(0) # (77,64,64)
            resized_map = resized_map + value if resized_map is not None else value
    elif timestep is None:
        for timestep_ in list(attn_maps.keys()):
            value = attn_maps[timestep_][path]
            value = torch.mean(value,axis=0).squeeze(0)
            seq_len, h, w = value.shape
            max_height = max(h, max_height)
            max_width = max(w, max_width)
            value = F.interpolate(
                value.to(dtype=torch.float32).unsqueeze(0),
                size=(max_height, max_width),
                mode='nearest',  
                # mode="bilinear", 
                # align_corners=False
            ).squeeze(0) # (77,64,64)
            resized_map = resized_map + value if resized_map is not None else value
    else:
        value = attn_maps[timestep][path]
        value = torch.mean(value,axis=0).squeeze(0)
        seq_len, h, w = value.shape
        max_height = max(h, max_height)
        max_width = max(w, max_width)
        value = F.interpolate(
            value.to(dtype=torch.float32).unsqueeze(0),
            size=(max_height, max_width),
            mode='nearest',   
            # mode='bilinear',
            # align_corners=False
        ).squeeze(0) # (77,64,64)
        resized_map = value

    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # init dirs
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if timestep is not None:
        save_path = save_path + f'/{timestep}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if path is not None:
        save_path = save_path + f'/{path}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    for i, (token, token_attn_map) in enumerate(zip(tokens, resized_map)):
        if token == bos_token:
            continue
        if token == eos_token:
            break
        token = token.replace('</w>','')
        token = f'{i}_<{token}>.jpg'

        # min-max normalization(for visualization purpose)
        token_attn_map = token_attn_map.numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)

        # save the image
        image = Image.fromarray(normalized_token_attn_map)
        image.save(os.path.join(save_path, token))


def save_by_timesteps_and_path(tokenizer, prompt, max_height, max_width, save_path, interpolation):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        for path in attn_maps[timestep].keys():
            resize_and_save(tokenizer, prompt, timestep, path, max_height=max_height, max_width=max_width, save_path=save_path, interpolation=interpolation)


def save_by_timesteps(tokenizer, prompt, max_height, max_width, save_path, interpolation):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        resize_and_save(tokenizer, prompt, timestep, None, max_height=max_height, max_width=max_width, save_path=save_path, interpolation=interpolation)


def save_by_path(tokenizer, prompt, max_height, max_width, save_path, interpolation):
    paths = attn_maps[list(attn_maps.keys())[0]]
    for path in tqdm(paths.keys(),total=len(list(attn_maps.keys()))):
        resize_and_save(tokenizer, prompt, None, path, max_height=max_height, max_width=max_width, save_path=save_path, interpolation=interpolation)