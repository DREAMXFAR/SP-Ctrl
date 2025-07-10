import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import cv2 
from pycocotools.coco import COCO
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import jsonlines
from typing import NamedTuple, List
# from mmpose.datasets.datasets.utils import parse_pose_metainfo

# from mmpose.registry import KEYPOINT_CODECS
import cv2

import math 

from loguru import logger

from torchvision import transforms
import torch

##################################################################################################
# Global Variables 
##################################################################################################
# get it from category_json["49"]["keypoints"]
KEYPOINT_NAME = [
    'left eye', 
    'right eye', 
    'nose', 
    'neck', 
    'root of tail', 
    'left shoulder', 
    'left elbow', 
    'left front paw', 
    'right shoulder', 
    'right elbow', 
    'right front paw', 
    'left hip', 
    'left knee', 
    'left back paw', 
    'right hip', 
    'right knee', 
    'right back paw', 
]



def save_kptname_embeddings():
    """save the text_emb of each keypoint for spatext-emb"""
    
    ###### AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "val"

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id), 
        "val": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id),
        "test": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id), 
    }
    
    json_path = dataset_split_info[split_mode]
    
    ##############################################################################################################
    ### get tokenizer and text_encoder from SD
    from transformers import AutoTokenizer, PretrainedConfig
    
    def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    tokenizer = AutoTokenizer.from_pretrained(
        r"/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(
        r"/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5", 
        False, 
    )
    text_encoder = text_encoder_cls.from_pretrained(
        r"/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5", 
        subfolder="text_encoder", 
        revision=False, 
    ).cuda()

    kpt_text_emb_dict = {}
    kpt_emb_list = []
    for akpt_name in KEYPOINT_NAME:
        tokenizer_inputs = tokenizer(akpt_name, max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt")
        tokenizer_inputs.to(text_encoder.device)
        _text_embeddings = text_encoder(**tokenizer_inputs).pooler_output
        
        kpt_text_emb_dict[akpt_name] = _text_embeddings.detach().cpu().squeeze().numpy()
        kpt_emb_list.append(_text_embeddings.detach().cpu())

        logger.info(f"{akpt_name}: {_text_embeddings.shape}")
        
    ##############################################################################################################
    # set save path
    kptemb_save_path = r"/dat03/xuanwenjie/code/controlnet/custom_datasets/kptname_emb"
    if not os.path.exists(kptemb_save_path):
        os.mkdir(kptemb_save_path)
    
    kpt_emb = torch.cat(kpt_emb_list, dim=0)

    # save_path = os.path.join(kptemb_save_path, "ap10k_kptname_emb.bin")
    # torch.save(kpt_emb, save_path)


if __name__ == "__main__":
    save_kptname_embeddings()