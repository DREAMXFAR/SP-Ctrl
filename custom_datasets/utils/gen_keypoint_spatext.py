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
from mmpose.datasets.datasets.utils import parse_pose_metainfo

from mmpose.registry import KEYPOINT_CODECS
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


##################################################################################################
# Utils Functions  
##################################################################################################
def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


def draw_bodypose(
        canvas: np.ndarray, 
        keypoints: List[Keypoint], 
        limbSeq, 
        only_keypoints: bool=False, 
    ) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.
    """
    stickwidth = 4

    ### for human 
    # limbSeq = [
    #     [2, 3], [2, 6], [3, 4], [4, 5], 
    #     [6, 7], [7, 8], [2, 9], [9, 10], 
    #     [10, 11], [2, 12], [12, 13], [13, 14], 
    #     [2, 1], [1, 15], [15, 17], [1, 16], 
    #     [16, 18],
    # ]

    # colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    #           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    colors = [(50+c, 50+c, 50+c) for c in range(1, 18)]

    if not only_keypoints:
        # draw skeletons
        for (k1_index, k2_index), color in zip(limbSeq, colors):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1 is None or keypoint2 is None:
                continue

            Y = np.array([keypoint1.x, keypoint2.x])   # * float(W)
            X = np.array([keypoint1.y, keypoint2.y])   # * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, [int(float(c)) for c in color])

    # draw keypoints
    # colors = [(10*c, 10*c, 10*c) for c in range(1, 18)]
    colors = [(c, c, c) for c in range(1, 18)]

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x)   # int(x * W)
        y = int(y)   # int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


class PILToTensor:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if len(target.shape) == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        return target


##################################################################################################
# main 
##################################################################################################
def save_spatext_map():
    """
    aim to save spatext map, not complete
    """
    ###### AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "train"
    black_background = True
    skeleton_style = "openpose"  # "mmpose"   # "openpose"

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id), 
        "val": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id),
        "test": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id), 
    }
    
    image_path = r"/dat03/xuanwenjie/datasets/AP-10K/data/"
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
    for akpt_name in KEYPOINT_NAME:
        tokenizer_inputs = tokenizer(akpt_name, max_length=tokenizer.model_max_length, padding="max_length", return_tensors="pt")
        tokenizer_inputs.to(text_encoder.device)
        _text_embeddings = text_encoder(**tokenizer_inputs).pooler_output
        kpt_text_emb_dict[akpt_name] = _text_embeddings.detach().cpu().squeeze().numpy()

        logger.info(f"{akpt_name}: {_text_embeddings.shape}")
        
    ##############################################################################################################

    ### load annotations 
    coco = COCO(annotation_file=json_path)

    ### class info
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    coco_supercats = {v["name"]: v["supercategory"] for k, v in coco.cats.items()}    

    # set save path
    heatmap_save_path = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/debug/spatext"
    if not os.path.exists(heatmap_save_path):
        os.mkdir(heatmap_save_path)

    # get all image index info
    ids = list(sorted(coco.imgs.keys()))
    print("==> number of images: {}".format(len(ids)))

    background_cats = set()

    pbar = tqdm(enumerate(coco.imgs.items()))
    for idx, (img_id, cur_img) in pbar:
        # info 
        pbar.set_description("[{}|{}]".format(idx, len(ids)))
        # get basic image information
        cur_img_id = cur_img["id"]
        cur_filename = cur_img["file_name"]
        cur_filename_wosuffix = os.path.splitext(cur_filename)[0]
        cur_width = cur_img["width"]
        cur_height = cur_img["height"]
        cur_background = cur_img["background"]

        if cur_background == 9 or cur_background == 88:
            print(cur_background, " -- ", cur_filename)

        background_cats.add(cur_background)

        # get annotation info
        # instance-seg
        cur_anns_ids = coco.getAnnIds(imgIds=[cur_img_id])
        cur_anns = coco.loadAnns(cur_anns_ids)

        assert cur_anns[0]["image_id"] == cur_img_id

        # save skeleton
        coco_skeleton = np.array(coco.loadCats(cur_anns[0]['category_id'])[0]['skeleton'])
        dataset_meta = parse_pose_metainfo(dict(from_file='/dat03/xuanwenjie/code/animal_pose/mmpose-main/configs/_base_/datasets/ap10k.py'))

        ### decoder settings 
        codec = dict(
            type='MSRAHeatmap', input_size=(cur_width, cur_height), heatmap_size=(64, 64), sigma=2)
        encoder = KEYPOINT_CODECS.build(codec)

        cur_img_path = os.path.join(image_path, cur_filename)
        cur_img_pil = Image.open(cur_img_path)
        cur_img = np.array(cur_img_pil)
        # save
        cur_img_pil.save(os.path.join(heatmap_save_path, "img.png"))
        
        ### get heatmap
        heatmap_array = np.zeros((17, 64, 64))

        for kid, ann in enumerate(cur_anns):
            kp = np.array(ann['keypoints'])

            keypoints = np.stack((kp[0::3], kp[1::3]), axis=1)[np.newaxis, :, :]
            keypoints_visible = (kp[2::3] > 0).astype(np.int32)[np.newaxis, :, ]

            encoded = encoder.encode(keypoints=keypoints, keypoints_visible=keypoints_visible)
            heatmap = encoded["heatmaps"]
            decoded = encoder.decode(encoded["heatmaps"])  

            ### vis individual heatmap 
            # grid_image_list = [Image.fromarray(np.uint8(heatmap[j, :, :]*255)) for j in range(17)]
            # grid_image = image_grid(grid_image_list, rows=3, cols=6)
            # grid_image.save(r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/debug/heatmap.png")

            heatmap_array += heatmap

        ### implementation-1: multiplication 
        heatmap_eye = heatmap_array[0]
        text_emb_eye = kpt_text_emb_dict["left eye"]
        output = heatmap_eye.reshape((64, 64, 1)) * text_emb_eye.reshape((1, 1, -1))

        ### implementation-2: torch.sparse
        import torch 
        shape = torch.Size([64, 64, 768])
        indices = [[15, 15, ], ]
        values = [text_emb_eye, ]
        sparse_matrix = torch.sparse_coo_tensor(indices, values, shape)



        save_path = os.path.join(heatmap_save_path, "{}.npy".format(cur_filename_wosuffix))
        # np.save(save_path, heatmap_array)

        ### check on raw image
        # raw_image_path = os.path.join(image_path, cur_filename)

        # raw_image_pil = Image.open(raw_image_path).convert("RGB")
        # vis_pil = Image.blend(raw_image_pil, heatmap_pil.convert("RGB"), alpha=0.7)
        # vis_pil.save(r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/debug/vis_heatmap_image.png")

        if idx > -1:
            break 

    return 0 


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

    save_path = os.path.join(kptemb_save_path, "ap10k_kptname_emb.bin")
    torch.save(kpt_emb, save_path)

   

def main():
    ### save spatext map
    save_spatext_map()

    ### prepare keypoint-name embedding
    # save_kptname_embeddings()


if __name__ == "__main__":

    main()