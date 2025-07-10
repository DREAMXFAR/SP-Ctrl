import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import numpy as np
from PIL import Image 
import json 
import cv2 
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from tqdm import tqdm
import argparse
import random 
from loguru import logger

"""
refer to /dat03/xuanwenjie/code/github_projects/freecontrol/myCode/utils/generate_frames_w_controlnet.py
"""
from utils import (
    attn_maps, 
    cross_attn_init,
    set_layer_with_name_and_path, 
    register_cross_attention_hook, 
    save_by_timesteps_and_path,
    save_by_timesteps, 
    save_by_path, 
)


##################################################################################################
# Global Variables 
##################################################################################################




##################################################################################################
# Utils Functions  
##################################################################################################
def image_grid(imgs, rows=2, cols=2, margin=10):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * (w+margin), rows * (h+margin)))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * (w+margin), i // cols * (h+margin)))
    
    return grid


def update_tokenizer(pipe, learned_embed_path, ):
    ### refer to textual-inversion: /dat03/xuanwenjie/code/textual_inversion/demo_diffusers/textual_inversion.py
    # Add the placeholder token in tokenizer, get keypoint names from category_json["49"]["keypoints"]
    kpt_token_dict = {
        1: {"keypoint_name": 'left eye', 
            "placeholder_token": "<left-eye>", "init_token": "eye", }, 
        2: {"keypoint_name": 'right eye', 
            "placeholder_token": "<right-eye>", "init_token": "eye", }, 
        3: {"keypoint_name": 'nose', 
            "placeholder_token": "<nose>", "init_token": "nose", }, 
        4: {"keypoint_name": 'neck', 
            "placeholder_token": "<neck>", "init_token": "neck", }, 
        5: {"keypoint_name": 'root of tail', 
            "placeholder_token": "<root-of-tail>", "init_token": "tail", }, 
        6: {"keypoint_name": 'left shoulder', 
            "placeholder_token": "<left-shoulder>", "init_token": "shoulder", }, 
        7: {"keypoint_name": 'left elbow', 
            "placeholder_token": "<left-elbow>", "init_token": "elbow", }, 
        8: {"keypoint_name": 'left front paw', 
            "placeholder_token": "<left-front-paw>", "init_token": "paw", }, 
        9: {"keypoint_name": 'right shoulder', 
            "placeholder_token": "<right-shoulder>", "init_token": "shoulder", }, 
        10: {"keypoint_name": 'right elbow', 
                "placeholder_token": "<right-elbow>", "init_token": "elbow", }, 
        11: {"keypoint_name": 'right front paw', 
                "placeholder_token": "<right-front-paw>", "init_token": "paw", }, 
        12: {"keypoint_name": 'left hip', 
                "placeholder_token": "<left-hip>", "init_token": "hip", }, 
        13: {"keypoint_name": 'left knee', 
                "placeholder_token": "<left-knee>", "init_token": "knee", }, 
        14: {"keypoint_name": 'left back paw', 
                "placeholder_token": "<left-back-paw>", "init_token": "paw", }, 
        15: {"keypoint_name": 'right hip', 
                "placeholder_token": "<right-hip>", "init_token": "hip", }, 
        16: {"keypoint_name": 'right knee', 
                "placeholder_token": "<right-knee>", "init_token": "knee", }, 
        17: {"keypoint_name": 'right back paw', 
                "placeholder_token": "<right-back-paw>", "init_token": "paw", }, 
    }

    placeholder_tokens = [atoken["placeholder_token"] for atoken in kpt_token_dict.values()]
    initializer_tokens = [atoken["init_token"] for atoken in kpt_token_dict.values()]

    num_added_tokens = pipe.tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != len(placeholder_tokens):
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_tokens}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # load pretrained token_embs
    pre_learned_emb = torch.load(learned_embed_path, )

    for aph_token, ainit_token in zip(placeholder_tokens, initializer_tokens):
        # Convert the initializer_token, placeholder_token to ids
        token_ids = pipe.tokenizer.encode(ainit_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = [token_ids[0]]
        placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids([aph_token])

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_id:
                token_embeds[token_id] = pre_learned_emb[aph_token].unsqueeze(0).clone()
            
    return pipe


def controlnet_infer(
        base_model_path, controlnet_path, control_image_path, 
        prompt, negative_prompt, 
        output_root, save_name, seed=555, 
        control_guidance_start=0.0,   # by xwj 231024
        control_guidance_end=1.0, 
        fixed_size=True, 
        raw_img_path=None, 
        learned_embed_path=None, 
    ):
    
    ################################################################################
    # add cross-attention implementation
    # refer to: https://github.com/wooyeolBaek/attention-map
    cross_attn_init()

    ################################################################################
    logger.info(" model_path: {}".format(controlnet_path))

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)  # , low_cpu_mem_usage=False)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.safety_checker = None
    pipe = pipe.to("cuda")

    ################################################################################
    # update tokenizer
    if not learned_embed_path is None:
        pipe = update_tokenizer(pipe, learned_embed_path)
        logger.info("load learned embedding: {}".format(learned_embed_path))

    ################################################################################
    # add cross-attention implementation
    pipe.unet = set_layer_with_name_and_path(pipe.unet)
    pipe.unet = register_cross_attention_hook(pipe.unet)

    pipe.controlnet = set_layer_with_name_and_path(pipe.controlnet)
    pipe.controlnet = register_cross_attention_hook(pipe.controlnet, prefix="controlnet")

    ################################################################################
    orig_control_image = load_image(control_image_path)

    if fixed_size:
        image_shape = (512, 512)
    else:
        image_shape = orig_control_image.size 
    control_image = orig_control_image.resize((512, 512), Image.BILINEAR)

    logger.info(" prompt: {}".format(prompt))

    # generate image
    logger.info(" seed: {}".format(seed))
    generator = torch.manual_seed(seed)
    
    """ hyper-parameters
    num_inference_stept = 50
    UniPCMultistepScheduler
    guidance_scale = 7.5
    eta = 0
    controlnet_condition_scalt = 1.0
    guess_mode = False
    control_guidance_start = 0.0, control_guidance_end = 1.0
    """
    output_image = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=10, 
            generator=generator, 
            image=control_image, 
            num_images_per_prompt=1, 
            control_guidance_start=control_guidance_start, 
            control_guidance_end=control_guidance_end, 
            # controlnet_conditioning_scale=0.0, 
        ).images
    
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    
    save_path = os.path.join(output_root, save_name)
    ### save mixed images 
    output_image.insert(0, orig_control_image)
    if raw_img_path is not None:
        output_image.insert(0, Image.open(raw_img_path))
    
    # resize images 
    output_image = [aimg.resize(image_shape, Image.BILINEAR) for aimg in output_image]

    # blend images 
    output_image = [Image.blend(aimg, control_image, 0.0) for aimg in output_image]   # 0.9 is all control image
    
    mix_image = image_grid(output_image, rows=1, cols=3)
    mix_image.save(save_path)    
    logger.info(" save_path: {}".format(save_path))


    ################################################################################
    # add cross-attention implementation
    ##### 3. Process and Save attention map #####
    height, width = image_shape
    logger.info('resizing and saving ...')

    # #### 3-1. save by timesteps and path (2~3 minutes) #####
    # interpolation_mode = "bilinear"
    # logger.info(" 3-1. save by timesteps and path: mode={}".format(interpolation_mode))
    # save_path = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/cross-attention-vis/attn_maps_by_timesteps_path/{}".format(save_name.split(".")[0])
    # save_by_timesteps_and_path(pipe.tokenizer, prompt, height, width, save_path, interpolation_mode)
    ###############################################################################

    ### 3-2. save by timesteps (1~2 minutes) #####
    interpolation_mode = "bilinear"
    logger.info("3-2. save by timesteps: mode={}".format(interpolation_mode))
    save_path = r"/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/cross-attention-vis/attn_maps_by_timesteps/{}".format(save_name.split(".")[0])
    save_by_timesteps(pipe.tokenizer, prompt, height, width, save_path, interpolation_mode)
    ###############################################################################

    # #### 3-3. save by path (1~2 minutes) #####
    # # TODO:    
    # interpolation_mode = "bilinear"
    # logger.info("3-3. save by path: mode={}".format(interpolation_mode))
    # save_path = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/cross-attention-vis/attn_maps_by_path/{}".format(save_name.split(".")[0])
    # save_by_path(pipe.tokenizer, prompt, height, width, save_path, interpolation_mode)
    ################################################################################


def infer_single():
    # set random seed
    seed = 42

    ### model setting 
    base_model_path = "/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5"

    # controlnet_path = r"/dat04/xuanwenjie/supercom/files/sc_controlnet_sd_ap10k_keypoint_crossattnloss_partition_kpttoken_nomseloss_downblocks_1/checkpoint-240000/controlnet/"
    controlnet_path = r"/dat04/xuanwenjie/supercom/files/sc_controlnet_sd_ap10k_keypoint_crossattnloss_partition_kpttoken_mseloss_detachq/checkpoint-240000/controlnet/"

    tmp_learned_embed_path = os.path.join(controlnet_path, "../learned_embeds.bin")
    if os.path.isfile(tmp_learned_embed_path):
        learned_embed_path = tmp_learned_embed_path
    else:
        learned_embed_path = None

    ### output setting
    output_root = "/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/cross-attention-vis/"
    if not os.path.exists(output_root):
        os.mkdir(output_root)

     #############################################################################################
    ### inputs 
    example_dict = {
        "demo_1": {
            "filename": "000000001996", 
            "prompt": "a photo of buffalo with left eye and right eye, left leg and right leg in the desert.", 
        }, 
        "demo_2": {
            "filename": "000000001996", 
            "prompt": "a photo of buffalo <<left-eye>, <right-eye>, <nose>, <neck>, <root-of-tail>, <left-shoulder>, <left-elbow>, <left-front-paw>, <left-knee>, <right-shoulder>, <right-elbow>, <right-front-paw>, <left-hip>, <left-knee>, <left-back-paw>, <right-hip>, <right-knee>, <right-back-paw>>", 
        }, 
    }

    demo_id = "demo_2"
    #############################################################################################
    
    filename = example_dict[demo_id]["filename"]
    # control_image_path = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/val_split1_openpose_skeleton/{}.png".format(filename) 
    control_image_path = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/pose_kptindex/{}.png".format(filename) 

    prompt = example_dict[demo_id]["prompt"]
    negative_prompt = ""
    print("==> filename = {}".format(filename))

    ### other settings
    raw_img_root = r"/dat03/xuanwenjie/datasets/AP-10K/data/"
    
    if prompt == "":
        save_name = "vis_{}_s_{}_woprompt.png".format(filename, seed)
    else:
        save_name = "vis_{}_s_{}_p_{}.png".format(filename, seed, prompt[:20])

    # run inference
    controlnet_infer(
            base_model_path, 
            controlnet_path,
            control_image_path,
            prompt,
            negative_prompt=negative_prompt, 
            output_root=output_root, 
            save_name=save_name,
            seed=seed, 
            raw_img_path=os.path.join(raw_img_root, filename + ".jpg"), 
            learned_embed_path=learned_embed_path, 
        )


if __name__ == "__main__":
    
    # inference 
    infer_single()

    print(":) Congratulations!")
