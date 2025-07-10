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

import zipfile
from io import BytesIO
from torchvision import transforms
import re 

import sys
sys.path.append(r"/dat03/xuanwenjie/code/controlnet/exp")
from tmp_src import SpaTextControlNetModel, StableDiffusionSpaTextControlNetPipeline

from loguru import logger

"""
refer to /dat03/xuanwenjie/code/github_projects/freecontrol/myCode/utils/generate_frames_w_controlnet.py
"""

##################################################################################################
# Global Variables 
##################################################################################################
DEBUG = False 

KEYPOINT_NAME = [
    "nose",
    "left eye", 
    "right eye", 
    "left ear", 
    "right ear", 
    "left shoulder", 
    "right shoulder", 
    "left elbow",
    "right elbow", 
    "left wrist",
    "right wrist", 
    "left hip", 
    "right hip", 
    "left knee", 
    "right knee", 
    "left ankle", 
    "right ankle",   
]

NUM_KEYPOINTS = 17

##################################################################################################
# Utils Functions  
##################################################################################################
def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class TestPipeline(object):
    def __init__(self, seed=555):
        self.pipe = None

        # set random seed
        random.seed(42)
        np.random.seed(66)
        self.generator = torch.manual_seed(seed)
        print("[Test] -- init torch-generator seed: {}".format(seed))

    def init_controlnet(self, controlnet_path, base_model_path, controlnet_cls=None, pipeline_cls=None, 
                        spatext_skeleton_type="one"): 
        
        kwargs = {}
        if not controlnet_cls is None:
            controlnet_cls = controlnet_cls
            kwargs["spatext_init_emb"] = None  #  while the __init__ is None, load the parameters from_pretrained, where self.kpt_emb is right
            kwargs["spatext_skeleton_type"] = spatext_skeleton_type
        else:
            controlnet_cls = ControlNetModel

        if not pipeline_cls is None:
            pipeline_cls = pipeline_cls
        else:
            pipeline_cls = StableDiffusionControlNetPipeline
        
        controlnet = controlnet_cls.from_pretrained(controlnet_path, torch_dtype=torch.float16, **kwargs)  # , low_cpu_mem_usage=False)
        self.pipe = pipeline_cls.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

        ### info 
        logger.info("conditioning_channels: {}".format(self.pipe.controlnet.conditioning_channels))
        if isinstance(controlnet, SpaTextControlNetModel):
            logger.info("skeleton_type: {}".format(self.pipe.controlnet.config.spatext_skeleton_type))
            logger.info("keypoint_init_path (when loading params, it is overwrited.): {}".format(self.pipe.controlnet.spatext_embedding.kpt_emb_path))  # stored in config.json
            
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.safety_checker = None
        print("[Test] -- init controlnet with checkpoints: {}".format(controlnet_path))


    def update_tokenizer(self, learned_embed_path, ):
        ### refer to textual-inversion: /dat03/xuanwenjie/code/textual_inversion/demo_diffusers/textual_inversion.py
        # Add the placeholder token in tokenizer, get keypoint names from category_json["49"]["keypoints"]
        kpt_token_dict = {
            1: {"keypoint_name": 'nose', 
                "placeholder_token": "<nose>", "init_token": "nose", }, 
            2: {"keypoint_name": 'left eye', 
                "placeholder_token": "<left-eye>", "init_token": "eye", }, 
            3: {"keypoint_name": 'right eye', 
                "placeholder_token": "<right-eye>", "init_token": "eye", }, 
            4: {"keypoint_name": 'left ear', 
                "placeholder_token": "<left-ear>", "init_token": "ear", }, 
            5: {"keypoint_name": 'right ear', 
                "placeholder_token": "<right-ear>", "init_token": "ear", }, 
            6: {"keypoint_name": 'left shoulder', 
                "placeholder_token": "<left-shoulder>", "init_token": "shoulder", }, 
            7: {"keypoint_name": 'right shoulder', 
                "placeholder_token": "<right-shoulder>", "init_token": "shoulder", }, 
            8: {"keypoint_name": 'left elbow', 
                "placeholder_token": "<left-elbow>", "init_token": "elbow", }, 
            9: {"keypoint_name": 'right elbow', 
                "placeholder_token": "<right-elbow>", "init_token": "elbow", }, 
            10: {"keypoint_name": 'left wrist', 
                 "placeholder_token": "<left-wrist>", "init_token": "wrist", }, 
            11: {"keypoint_name": 'right wrist', 
                 "placeholder_token": "<right-wrist>", "init_token": "wrist", }, 
            12: {"keypoint_name": 'left hip', 
                 "placeholder_token": "<left-hip>", "init_token": "hip", }, 
            13: {"keypoint_name": 'right hip', 
                 "placeholder_token": "<right-hip>", "init_token": "hip", }, 
            14: {"keypoint_name": 'left knee', 
                 "placeholder_token": "<left-knee>", "init_token": "knee", }, 
            15: {"keypoint_name": 'right knee', 
                 "placeholder_token": "<right-knee>", "init_token": "knee", }, 
            16: {"keypoint_name": 'left ankle', 
                 "placeholder_token": "<left-ankle>", "init_token": "ankle", }, 
            17: {"keypoint_name": 'right ankle', 
                 "placeholder_token": "<right-ankle>", "init_token": "ankle", }, 
        }

        placeholder_tokens = [atoken["placeholder_token"] for atoken in kpt_token_dict.values()]
        initializer_tokens = [atoken["init_token"] for atoken in kpt_token_dict.values()]

        num_added_tokens = self.pipe.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_tokens}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        
        # load pretrained token_embs
        pre_learned_emb = torch.load(learned_embed_path, )

        for aph_token, ainit_token in zip(placeholder_tokens, initializer_tokens):
            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.pipe.tokenizer.encode(ainit_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = [token_ids[0]]
            placeholder_token_id = self.pipe.tokenizer.convert_tokens_to_ids([aph_token])

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = self.pipe.text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in placeholder_token_id:
                    token_embeds[token_id] = pre_learned_emb[aph_token].unsqueeze(0).clone()


    def inference_image(
            self, 
            control_image_path, 
            prompt, 
            output_root, 
            seed=555, 
            negative_prompt="", 
            control_guidance_start=0.0, 
            control_guidance_end=1.0, 
            controlnet_conditioning_scale=1.0, 
            fixed_size=True, 
            raw_img_path=None, 
        ):
        # set image 
        image_name_wosuffix = os.path.splitext(os.path.basename(control_image_path))[0]
        orig_control_image = load_image(control_image_path)

        if fixed_size:
            image_shape = (512, 512)
        else:
            image_shape = orig_control_image.size 
        control_image = orig_control_image.resize((512, 512), Image.BILINEAR)

        # set generator
        print("==> seed: {}".format(seed))
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
        output_image = self.pipe(
                prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=50, 
                generator=generator, 
                image=control_image, 
                num_images_per_prompt=4, 
                control_guidance_start=control_guidance_start, 
                control_guidance_end=control_guidance_end, 
                controlnet_conditioning_scale=controlnet_conditioning_scale, 
            ).images
        
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        
        ### save mixed images 
        output_image.insert(0, orig_control_image)
        if raw_img_path is not None:
            output_image.insert(0, Image.open(raw_img_path))
        
        # resize images 
        output_image = [aimg.resize(image_shape, Image.BILINEAR) for aimg in output_image]
        # blend images 
        output_image = [Image.blend(aimg, control_image, 0.45) for aimg in output_image]
        
        save_path = os.path.join(output_root, image_name_wosuffix + ".png")
        mix_image = image_grid(output_image, rows=1, cols=5)
        mix_image.save(save_path)    
        print("[Test] -- save image in: {}".format(save_path))


    def inference_image_spatext(
            self, 
            control_image_path, 
            prompt, 
            output_root, 
            seed=555, 
            negative_prompt="", 
            raw_image_path=None, 
            control_guidance_start=0.0, 
            control_guidance_end=1.0, 
            controlnet_conditioning_scale=1.0, 
            fixed_size=False, 
            woprompt_on=False, 
        ):
        # info 
        image_name = os.path.basename(control_image_path)
        image_name_wosuffix = os.path.splitext(image_name)[0]
        
        ### load control images 
        num_images_per_prompt = 3
        
        # prepare prompts
        if woprompt_on:
            cur_caption = ""
        else:
            cur_caption = prompt
            cur_negative_caption = negative_prompt
        
        # load images 
        cur_raw_image = load_image(raw_image_path).convert("RGB")
        cur_image_shape = cur_raw_image.size

        if fixed_size:
            image_shape = (512, 512)
        else:
            image_shape = cur_image_shape 

        # prepare and load images 
        def load_spatext_img(cond_path):
            image_pil = Image.open(cond_path).convert("L")
            image_tensor = torch.as_tensor(np.array(image_pil), dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        cur_control_image = load_spatext_img(control_image_path)
        resize_func = transforms.Resize((512, 512), antialias=True, interpolation=Image.NEAREST)
        cur_resize_control_image = resize_func(cur_control_image)

        # use individual generator 
        generator = torch.Generator("cuda").manual_seed(seed)

        """ 
        hyper-parameters
            num_inference_stept = 50
            UniPCMultistepScheduler
            guidance_scale = 7.5
            eta = 0
            controlnet_condition_scalt = 1.0
            guess_mode = False
            control_guidance_start = 0.0, control_guidance_end = 1.0
        """
        output_image = self.pipe(
            cur_caption, 
            negative_prompt=negative_prompt, 
            num_inference_steps=50, 
            generator=generator, 
            image=cur_resize_control_image, 
            num_images_per_prompt=num_images_per_prompt, 
            uidance_scale=7.5, 
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
            control_guidance_start=control_guidance_start, 
            control_guidance_end=control_guidance_end, 
            add_kpttoken_from=None, 
            add_kpttoken_to=None, 
        ).images

        ### save mixed images 
        control_image = Image.open(control_image_path).convert("RGB")
        output_image.insert(0, control_image)
        if raw_image_path is not None:
            output_image.insert(0, Image.open(raw_image_path))
        
        # resize images 
        output_image = [aimg.resize(image_shape, Image.BILINEAR) for aimg in output_image]
        # blend images 
        output_image = [Image.blend(aimg, control_image, 0.45) for aimg in output_image]
        
        save_path = os.path.join(output_root, image_name_wosuffix + ".png")
        mix_image = image_grid(output_image, rows=1, cols=5)
        mix_image.save(save_path)    
        print("[Test] -- save image in: {}".format(save_path))

        return 0


def arg_parser():
    parser = argparse.ArgumentParser(description='Get conditional inputs.')     
    parser.add_argument('--controlnet_path', type=str, default=None)
    parser.add_argument('--control_image_path', type=str, default=None)
    parser.add_argument('--raw_image_path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--cond_mode', type=str, default="ap_spatext")
    parser.add_argument('--enable_learnable_token', action="store_true",)
    parser.add_argument('--spatext_skeleton_type', type=str, default="one",)

    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    
    ### config
    seed = 66

    ### get augments 
    args = arg_parser()
    
    # other configs
    if DEBUG:
        ### ours
        args.cond_mode = "ap_spatext"
        args.spatext_skeleton_type = "one"
        args.add_kpt_prompt = True
        args.enable_learnable_token = True
        
        args.control_image_path = r"/dat03/xuanwenjie/datasets/HumanArt_triplet/humanart_kptindex/real_human/dance/000000000274.png"
        
        args.raw_image_path = r"/dat03/xuanwenjie/datasets/HumanArt/images/real_human/dance/000000000274.jpg"
        # args.prompt = "dance, two dancers on the stage"
        args.prompt = "dance, two dancers on the stage. <nose>, <left-eye>, <right-eye>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>"
        
        args.negative_prompt = ""
        
        # VALIDATION_IMAGE_1='/dat99/priv/jhliu4/xuanwenjie/datasets/HumanArt_triplet/humanart_kptindex/real_human/dance/000000000274.png'
        # VALIDATION_PROMPT_1='dance, two dancers in gold costumes performing on stage. <nose>, <left-eye>, <right-eye>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>'
        # VALIDATION_IMAGE_2='/dat99/priv/jhliu4/xuanwenjie/datasets/HumanArt_triplet/humanart_kptindex/real_human/dance/000000000427.png'
        # VALIDATION_PROMPT_2='dance, a female dancer performing on stage in front of a blue light. <nose>, <left-eye>, <right-eye>, <left-ear>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>'
        # VALIDATION_IMAGE_3='/dat99/priv/jhliu4/xuanwenjie/datasets/HumanArt_triplet/humanart_kptindex/real_human/dance/000000001406.png'
        # VALIDATION_PROMPT_3='dance, two women in costumes. <nose>, <left-eye>, <right-eye>, <left-ear>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>'

        print("==> Warning: redundant settings!")
 
    if args.output_root is None: 
        args.output_root = r"/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/demo" 
    
    if args.controlnet_path is None: 
        args.controlnet_path = r"/dat04/xuanwenjie/supercom/tmp/transfer_dirs/sc_controlnet_sd_humanart_keypoint_crossattnloss_partition_kpttoken_mseloss_timestep_250_500_downblocks_2_detachq_spatext_kptrand_sksone/checkpoint-528000/controlnet"
    
    ### preprocess
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    ### basic settings
    base_model_path = "/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5"
    controlnet_path = args.controlnet_path

    test_pipeline = TestPipeline()

    if args.cond_mode == "ap_spatext": 
        test_pipeline.init_controlnet(
            controlnet_path=controlnet_path,
            base_model_path=base_model_path,  
            controlnet_cls=SpaTextControlNetModel, 
            pipeline_cls=StableDiffusionSpaTextControlNetPipeline, 
            spatext_skeleton_type=args.spatext_skeleton_type, 
        )
    else:
        test_pipeline.init_controlnet(
                controlnet_path=controlnet_path,
                base_model_path=base_model_path,  
            )
    
    if args.enable_learnable_token:
        learned_embed_path = os.path.join(controlnet_path, "../learned_embeds.bin")
        test_pipeline.update_tokenizer(learned_embed_path)
    
    if args.cond_mode in ["ap", "ap_samhqmask", "ap_zeodepth"]:
        # info
        print("==> load learned tokens: {}".format(args.enable_learnable_token))

        test_pipeline.inference_image(
            control_image_path = args.control_image_path, 
            prompt = args.prompt, 
            negative_prompt = args.negative_prompt, 
            raw_img_path = args.raw_image_path, 
            output_root=args.output_root, 
            seed=seed, 
        )
    elif args.cond_mode == "ap_spatext":
        # info
        print("==> load learned tokens: {}".format(args.enable_learnable_token))
        
        ### batched inference with images
        test_pipeline.inference_image_spatext(
            control_image_path=args.control_image_path, 
            raw_image_path=args.raw_image_path, 
            prompt=args.prompt, 
            output_root=args.output_root, 
            seed=seed, 
            negative_prompt=args.negative_prompt, 
        ) 
    else:
        NotImplementedError        

    print(":) Congratulations!")
