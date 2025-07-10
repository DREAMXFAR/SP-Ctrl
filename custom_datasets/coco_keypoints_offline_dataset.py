import json
import cv2
import numpy as np
# from torch.utils.data import Dataset
import os
import datasets
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

import pycocotools.mask as mask_utils
import re


######################################################################################################################################
# Information
######################################################################################################################################
### dataset root
DATASET_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/"

### get category json 
category_json_path = os.path.join(DATASET_ROOT, r"catgories.jsonl")
with open(category_json_path, 'r') as f:
    category_json = json.load(f)

### set random seed 
random.seed(42)

category_names = [c["name"].split(" ")[-1] for c in category_json.values()]

name_token_mapping = {
    "alouatta": "atta", 
    "uakari": "ari", 
    "orangutan": "tan", 
    "marmot": "mot", 
}

for idx, aname in enumerate(category_names):
    if aname in name_token_mapping.keys():
        category_names[idx] = name_token_mapping[aname]
    

BACKGROUND_CATEGORY = {
    "1": "grass or savanna",
    "2": "forest or shrub", 
    "3": "mud or rock", 
    "4": "snowfield", 
    "5": "zoo or human habitation", 
    "6": "swamp or riverside", 
    "7": "desert or gobi", 
    "8": "mugshot", 
    "9": "forest",   # 000000039789.jpg
    "88": "dark",  # 000000050019.jpg
}

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

KEYPOINT_PLACEHOLDER = [
    '<left-eye>', 
    '<right-eye>', 
    '<nose>', 
    '<neck>', 
    '<root-of-tail>', 
    '<left-shoulder>', 
    '<left-elbow>', 
    '<left-front-paw>', 
    '<right-shoulder>', 
    '<right-elbow>', 
    '<right-front-paw>', 
    '<left-hip>', 
    '<left-knee>', 
    '<left-back-paw>', 
    '<right-hip>', 
    '<right-knee>', 
    '<right-back-paw>', 
]

keypoint_names = [c.split(" ")[-1] for c in KEYPOINT_NAME]

NUM_KEYPOINTS = 17

######################################################################################################################################
# Functions 
######################################################################################################################################
class PILToTensor:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        return target


######################################################################################################################################
# Class
######################################################################################################################################
class AP10kOfflineDataset():
    """
    The condition images are offline generated.  
    """
    def __init__(self, args=None, tokenizer=None, text_encoder=None, split='val', 
                 return_objname_pos=False, extra_cond=None, add_kpt_prompt=0.0, use_heatmap=False, ):
        ### mode in: openpose, mmpose, openpose_catonly, keypoint
        self.conditioning_mode = args.conditioning_mode
        if self.conditioning_mode == "keypoint":
            ### set path
            self.json_path = os.path.join(DATASET_ROOT, r"{}_split1.jsonl".format(split))
            self.image_root = os.path.join(DATASET_ROOT, r'data')
            self.conditioning_root = os.path.join(DATASET_ROOT, "only_keypoint", r'{}_split1_openpose_keypoint'.format(split))
        elif self.conditioning_mode == "spatext":
            ### set path
            self.json_path = os.path.join(DATASET_ROOT, r"{}_split1.jsonl".format(split))
            self.image_root = os.path.join(DATASET_ROOT, r'data')
            self.conditioning_root = os.path.join(DATASET_ROOT, "pose_kptindex")
        elif self.conditioning_mode == "sam_hq_mask":
            ### set path
            self.json_path = os.path.join(DATASET_ROOT, r"{}_split1.jsonl".format(split))
            self.image_root = os.path.join(DATASET_ROOT, r'data')
            self.conditioning_root = os.path.join(DATASET_ROOT, "sam_hq_mask")
        elif self.conditioning_mode == "zeo_depth_masked":
            ### set path
            self.json_path = os.path.join(DATASET_ROOT, r"{}_split1.jsonl".format(split))
            self.image_root = os.path.join(DATASET_ROOT, r'data')
            self.conditioning_root = os.path.join(DATASET_ROOT, "zeo_depth_masked")
        else:
            ### set path
            self.json_path = os.path.join(DATASET_ROOT, r"{}_split1.jsonl".format(split))
            self.image_root = os.path.join(DATASET_ROOT, r'data')
            self.conditioning_root = os.path.join(DATASET_ROOT, r'{}_split1_{}_skeleton'.format(split, self.conditioning_mode))

        ### 2409, deal with extra conditional images 
        self.extra_cond = extra_cond
        if not self.extra_cond is None:
            self.extra_cond_root = os.path.join(DATASET_ROOT, extra_cond)

        ### 2409 add heatmap
        self.use_heatmap = use_heatmap
        self.heatmap_type = args.heatmap_type
        if self.use_heatmap:
            if self.heatmap_type == "whole":
                self.heatmap_root = os.path.join(DATASET_ROOT, r"heatmap_stacked")
            elif self.heatmap_type == "partition":
                self.heatmap_root = os.path.join(DATASET_ROOT, r"heatmap_multichannel_npy")

        # 230922 by xwj, adapt to GLIGEN grounding token 
        self.return_objname_pos = return_objname_pos
        
        if split == 'train':
            self.is_train = True
        elif split == 'val':
            self.is_train = False
        else:
            raise Exception("The split {} is not defined.".format(split))

        self.args = args
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder  # not used 
        self.column_names = []
        self.add_kpt_prompt = add_kpt_prompt  # the percentage of adding keypoint name to prompt
        self.add_cross_attn_loss = args.add_cross_attn_loss

        self.data = []
        with open(self.json_path, 'r') as f:
            json_data = f.readlines()
            for line in json_data:
                self.data.append(json.loads(line))


    def __len__(self):
        return len(self.data)


    def tokenize_captions(self, caption):
        # sometimes use non-prompt, noted by xwj
        if random.random() < self.args.proportion_empty_prompts: 
            caption = ""
        elif isinstance(caption, str):
            caption = caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption = random.choice(caption) if self.is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        ### get place_holder
        token_placeholder_mask = None
        if self.add_cross_attn_loss:   
            token_placeholder_mask = np.zeros(self.tokenizer.model_max_length)
            
            if caption == "":
                pass
            else:      
                bos_token = self.tokenizer.bos_token
                eos_token = self.tokenizer.eos_token
                pad_token = self.tokenizer.pad_token

                if self.heatmap_type == "whole":
                    token_names = category_names
                elif self.heatmap_type == "partition":
                    token_names = KEYPOINT_PLACEHOLDER if self.args.enable_learnable_kpttoken else keypoint_names
                
                for i, token_id in enumerate(inputs.input_ids[0]):
                    cur_token = self.tokenizer.decode(token_id)
                    if cur_token == bos_token:
                        continue
                    if cur_token == eos_token:
                        break
                    
                    if cur_token in token_names:
                        token_placeholder_mask[i] = 1
                        
                # print(f"{caption} -- {token_placeholder_mask}")

        return inputs.input_ids[0], token_placeholder_mask


    def preprocess(
            self, image, text, conditioning_image, object_name, center_pos, 
            sample_object_pos_mode="default", extra_conditioning_image=None, heatmap=None, keypoint_prompt_vis_mask=None):
        
        # get basic information 
        orig_w, orig_h = image.size 

        # preprocess image
        image_transforms = transforms.Compose(
            [   
                ### original implementation 
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), 
            ]
        )
        image = image_transforms(image)
        
        # preprocess cond images
        if self.conditioning_mode in ["spatext"]:
            conditioning_image_transforms = transforms.Compose([
                    transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.CenterCrop(self.args.resolution),
                    PILToTensor(),
                ])
        else:
            conditioning_image_transforms = transforms.Compose([
                    transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self.args.resolution),
                    transforms.ToTensor(),
                ])
        conditioning_image = conditioning_image_transforms(conditioning_image)

        # 2409 by xwj
        extra_conditions = {}
        if not extra_conditioning_image is None:
            # max=1.0, min=0
            extra_conditioning_image = conditioning_image_transforms(extra_conditioning_image)
            conditioning_image = torch.cat([conditioning_image, extra_conditioning_image], dim=0)

        other_args = {}
        if self.use_heatmap and heatmap is not None:
            # 2409 by xwj, deal with heatmap 
            if isinstance(heatmap, list):
                heatmap_list = []
                for ahm in heatmap:
                    ahm = conditioning_image_transforms(ahm)
                    heatmap_list.append(ahm)
                heatmap_image = torch.concat(heatmap_list, dim=0)
            else:    
                heatmap_image = conditioning_image_transforms(heatmap)
            other_args["heatmap"] = heatmap_image
 
        # tokenize
        if self.tokenizer is None:
            text = text
        else: 
            text, token_placeholder_mask = self.tokenize_captions(text)
            # only self.add_cross_attn_loss, token_place_holder_mask is not None
            other_args["token_placeholder_mask"] = torch.from_numpy(token_placeholder_mask) if not token_placeholder_mask is None else None

        if self.add_cross_attn_loss:
            if token_placeholder_mask.sum() == 0:
                other_args["keypoint_prompt_vis_mask"] = torch.zeros(17).to(torch.bool)
            else:
                other_args["keypoint_prompt_vis_mask"] = torch.from_numpy(keypoint_prompt_vis_mask)
            
            token_mask_sum = torch.sum(other_args["token_placeholder_mask"])
            kpt_prompt_vis_mask_sum = torch.sum(other_args["keypoint_prompt_vis_mask"])
            if self.heatmap_type == "partition":
                # TODO: if the prompt is truncated, the keypoint visible mask should be truncated too.
                assert token_mask_sum == kpt_prompt_vis_mask_sum

        # process object name and center pos 
        grounding_tokens = {}

        return image, text, conditioning_image, grounding_tokens, other_args
    

    def get_object_name_from_anns(self, anns):
        object_name = []
        for cur_ann in anns:
            cur_category_id = cur_ann["category_id"]
            cur_category_name = category_json["{}".format(cur_category_id)]["name"]
            object_name.append(cur_category_name)
        return object_name


    def get_center_pos_from_anns(self, anns, ):
        center_pos = []
        for cur_ann in anns:
            cur_x, cur_y, cur_w, cur_h = cur_ann["bbox"] 
            cur_center_x = cur_x + cur_w/2.0
            cur_center_y = cur_y + cur_h/2.0

            center_pos.append([cur_center_x, cur_center_y])
        return np.array(center_pos)       


    def get_kptvisible_from_anns(self, anns, ):
        kpt_vis = np.zeros(NUM_KEYPOINTS)
        for cur_ann in anns:
            cur_kpt_vis = cur_ann["keypoints"][2::3]
            kpt_vis = kpt_vis + np.array(cur_ann["keypoints"][2::3])
        return kpt_vis > 0


    def __getitem__(self, idx):
        item = self.data[idx]
        ### get basic info 
        filename = item["filename"]
        filename_wosuffix = os.path.splitext(filename)[0]
        image_id = item["image_id"]
        height = item["height"] 
        width = item["width"] 
        anns = item["anns"] 
        background = BACKGROUND_CATEGORY[str(item["background"])]

        ### get image
        image_path = os.path.join(self.image_root, filename)
        image = Image.open(image_path).convert('RGB')

        ### get condition image
        conditioning_image = Image.open(os.path.join(self.conditioning_root, filename).replace('jpg', 'png'))
        if self.conditioning_mode == "spatext":
            conditioning_image = conditioning_image.convert("L")
        else:
            conditioning_image = conditioning_image.convert("RGB")

        ### get extra condition image
        extra_conditioning_image = None
        if not self.extra_cond is None:
            extra_conditioning_image = Image.open(os.path.join(self.extra_cond_root, filename).replace("jpg", "png")).convert("L")

        ### get heatmap 
        if self.use_heatmap:
            if self.heatmap_type == "whole":
                heatmap = Image.open(os.path.join(self.heatmap_root, filename).replace("jpg", "png")).convert("RGB")
            elif self.heatmap_type == "partition":
                heatmap_data = np.load(os.path.join(self.heatmap_root, filename.replace("jpg", "npy")))
                heatmap = []
                for hmid in range(NUM_KEYPOINTS):
                    cur_hm = Image.fromarray((255*heatmap_data[hmid, :, :]).astype(np.uint8))
                    cur_hm = cur_hm.resize((width, height))
                    heatmap.append(cur_hm)
        else:
            heatmap = None

        ### get image caption as prompt
        instance_cats = set([category_json[str(ann['category_id'])]["name"] for ann in anns])

        # refer to textual inversion and CLIP ImageNet classification
        caption_templates = [
            "A good photo of {}<KPT>".format(" ".join(instance_cats)), 
            "A photo of {}<KPT> in the {}".format(" ".join(instance_cats), background), 
            "There is {}<KPT> on the {}".format(" ".join(instance_cats), background), 
            "There are some {}<KPT> lying in the {}".format(" ".join(instance_cats), background), 
            "Some {}<KPT> are in the {}".format(" ".join(instance_cats), background), 
            "A close photo of {}<KPT>.".format(" ".join(instance_cats)), 
            "In the {}, there are severl {}<KPT>.".format(background, " ".join(instance_cats)), 
            "This is a clear photo of {}<KPT> in the {}.".format(" ".join(instance_cats), background), 
            "Several {}<KPT> are in the {}.".format(" ".join(instance_cats), background), 
            "A {}<KPT> stands in the {}.".format(" ".join(instance_cats), background),
        ]
        text = [random.choice(caption_templates)]

        # preprocess text
        tmp_text = text.copy()
        if random.random() < self.add_kpt_prompt:
            # collect gt vis
            kpt_vis = self.get_kptvisible_from_anns(anns=anns)  # (NUM_KEYPOINT) bool
            kpt_vis_list = []
            for avis, akptname in zip(kpt_vis, KEYPOINT_NAME):
                if self.args.enable_learnable_kpttoken:
                    akptname = "<" + "-".join(akptname.split(" ")) + ">"
                if avis:
                    kpt_vis_list.append(akptname) 
            keypoints_prompt = " <" + ", ".join(kpt_vis_list) + ">"
            text = [re.sub(r"<KPT>", keypoints_prompt, atxt) for atxt in tmp_text]

            keypoint_prompt_vis_mask = kpt_vis 
        else:
            text = [re.sub(r"<KPT>", "", atxt) for atxt in tmp_text]
            keypoint_prompt_vis_mask = np.zeros(17) > 0
        # print("==> text: {}".format(text))

        ### get object_name and position, 230923 
        object_name = None 
        center_pos = None 
        if self.return_objname_pos:
            object_name = self.get_object_name_from_anns(anns)
            center_pos = self.get_center_pos_from_anns(anns)

        if self.args is None:
            return dict(
                    filename=filename, pixel_values=image, input_ids=text, conditioning_pixel_values=conditioning_image, 
                )
        else:
            # NOTE: sample_object_pos_mode = "default": the first 30 objects may have the same category, "random": random sample 30
            image, text, conditioning_image, grounding_tokens, other_args = self.preprocess(
                                                                            image=image, 
                                                                            text=text, 
                                                                            conditioning_image=conditioning_image, 
                                                                            object_name=object_name, 
                                                                            center_pos=center_pos,
                                                                            sample_object_pos_mode="random", 
                                                                            extra_conditioning_image=extra_conditioning_image,
                                                                            heatmap=heatmap,  
                                                                            keypoint_prompt_vis_mask=keypoint_prompt_vis_mask, 
                                                                        )
            return dict(
                    file_name=filename, pixel_values=image, input_ids=text, conditioning_pixel_values=conditioning_image,
                    **grounding_tokens, **other_args, 
                )


######################################################################################################################################
# __main__
######################################################################################################################################
if __name__ == "__main__":
    from transformers import AutoTokenizer, PretrainedConfig
    from datasets import load_dataset
    import argparse
    
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
    

    parser = argparse.ArgumentParser(description='Demo')     
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--proportion_empty_prompts', type=float, default=0.0)
    parser.add_argument(
        "--conditioning_mode", type=str, default=None,
        help=( "Set the condition mode for custom offline dataset by xwj."),
    ) 

    args = parser.parse_args()
    # args.conditioning_mode = 'openpose_catonly'
    # args.conditioning_mode = 'keypoint'
    # args.conditioning_mode = 'spatext'
    args.conditioning_mode = 'sam_hq_mask'

    args.add_cross_attn_loss = False
    args.add_keypoint_prompt = 0
    args.heatmap_type = None  # "partition"
    args.use_heatmap = False

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
    )
    
    # canny  fakescribble  hed  midas  mlsd
    extra_cond = None  # "sam_hq_mask"
    train_dataset = AP10kOfflineDataset(args, split='val', tokenizer=tokenizer, return_objname_pos=False, 
                                        extra_cond=extra_cond, add_kpt_prompt=args.add_keypoint_prompt, use_heatmap=args.use_heatmap)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids = torch.stack([example["input_ids"] for example in examples])

        return_dict = {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
        }
         
        ### add additional key and values 
        # collect for mask, object_name and center pose 
        if "object_name_embedding" in examples[0].keys():
            # return_dict["object_name_embedding"] = torch.stack([example["object_name_embedding"] for example in examples])
            return_dict["object_name_embedding"] = [example["object_name_embedding"] for example in examples]
        if "center_pos" in examples[0].keys():    
            return_dict["center_pos"] = torch.stack([example["center_pos"] for example in examples])
        if "masks" in examples[0].keys():
            return_dict["masks"] = torch.stack([example["masks"] for example in examples])
        # other args 
        if "heatmap" in examples[0].keys():
            heatmap_pixel_values = torch.stack([example["heatmap"] for example in examples])
            heatmap_pixel_values = heatmap_pixel_values.to(memory_format=torch.contiguous_format).float()
            return_dict["heatmap"] = heatmap_pixel_values

        if "token_placeholder_mask" in examples[0].keys() and (not examples[0]["token_placeholder_mask"] is None):
            token_placeholder_mask = torch.stack([example["token_placeholder_mask"] for example in examples])
            token_placeholder_mask = token_placeholder_mask.to(memory_format=torch.contiguous_format).float()
            return_dict["token_placeholder_mask"] = token_placeholder_mask

        if "keypoint_visible_mask" in examples[0].keys():
            keypoint_visible_mask = torch.stack([example["keypoint_visible_mask"] for example in examples])
            keypoint_visible_mask = keypoint_visible_mask.to(memory_format=torch.contiguous_format).float()
            return_dict["keypoint_visible_mask"] = keypoint_visible_mask

        return return_dict


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn, 
        batch_size=1,
        num_workers=0,
    )

    ### detokenize
    detokenize_dict = tokenizer.decoder
    cnt = 0

    for idx, aitem in enumerate(train_dataloader):
        # info 
        print("{} -- {}".format(idx, type(aitem)))

        # conditioning_image in [0, 1]
        image = aitem["pixel_values"]
        conditioning_image = aitem["conditioning_pixel_values"]

        if "token_placeholder_mask" in aitem.keys():
            token_placeholder_mask = aitem["token_placeholder_mask"]
            # if not torch.sum(token_placeholder_mask) == 1:
            #     __import__("ipdb").set_trace()                

        if train_dataloader.batch_size == 1:
            cond_pil = conditioning_image.squeeze().numpy().transpose([1, 2, 0])
            cond_pil = np.ascontiguousarray(cond_pil * 255, dtype=np.uint8)
            w, h, c = cond_pil.shape 
            
            if train_dataset.return_objname_pos: 
                for i in range(30):
                    amask = aitem["masks"][0, i]

                    if amask == 1:
                        aobj_name = aitem["object_name_embedding"][0][i]
                        acenter_pos = aitem["center_pos"][0, i, :]

                        acenter_pos = acenter_pos * w 
                        ax = int(acenter_pos[0])
                        ay = int(acenter_pos[1])
                        cv2.putText(cond_pil, aobj_name, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cnt = cnt + 1
        if cnt > 1:
            break


"""debug
image = Image.fromarray(np.uint8(aitem["conditioning_pixel_values"].squeeze().numpy().transpose([1, 2, 0]))*255)
"""
