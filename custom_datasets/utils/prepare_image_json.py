import os 
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
import math 


def prepare_ap10k():
    ###### AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "train"
    skeleton_style = "openpose"  # "mmpose"   # "openpose"

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id), 
        "val": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id),
        "test": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id), 
    }
    
    image_path = r"/dat03/xuanwenjie/datasets/AP-10K/data/"
    json_path = dataset_split_info[split_mode]

    ### load annotations 
    coco = COCO(annotation_file=json_path)

    ### class info
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    coco_supercats = {v["name"]: v["supercategory"] for k, v in coco.cats.items()}    

    # set save path
    # SAVE_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/"
    SAVE_ROOT = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/debug/pose_json"

    # get all image index info
    ids = list(sorted(coco.imgs.keys()))
    print("==> number of images: {}".format(len(ids)))

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

        # get annotation info
        # instance-seg
        cur_anns_ids = coco.getAnnIds(imgIds=[cur_img_id])
        cur_anns = coco.loadAnns(cur_anns_ids)

        assert cur_anns[0]["image_id"] == cur_img_id

        # save skeleton
        coco_skeleton = np.array(coco.loadCats(cur_anns[0]['category_id'])[0]['skeleton'])
        dataset_meta = parse_pose_metainfo(dict(from_file='/dat03/xuanwenjie/code/animal_pose/mmpose-main/configs/_base_/datasets/ap10k.py'))
        # keypoint_id2name_dict = dataset_meta["keypoint_id2name"]

        # save triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "background": cur_background, 
        }

        # save one json for each image
        save_path = os.path.join(SAVE_ROOT, "{}.json".format(cur_filename_wosuffix))
        # print(save_path)
        with open(save_path, "w") as f:
            json.dump(cur_line, f, indent=4)

        # if idx > -1:
        #     break 

    return 0 


def prepare_humanart():
    ###### HumanArt
    ### cfg 
    split_mode = "train"
    skeleton_style = "openpose"  # "mmpose"   # "openpose"

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/HumanArt/annotations/training_humanart.json", 
        "val": r"/dat03/xuanwenjie/datasets/HumanArt/annotations/validation_humanart.json",
    }
    
    image_path = r"/dat03/xuanwenjie/datasets/HumanArt/images/"
    json_path = dataset_split_info[split_mode]

    ### load annotations 
    coco = COCO(annotation_file=json_path)

    ### class info
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    coco_supercats = {v["name"]: v["supercategory"] for k, v in coco.cats.items()}    

    # set save path
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/HumanArt_triplet/"

    # get all image index info
    ids = list(sorted(coco.imgs.keys()))
    print("==> number of images: {}".format(len(ids)))

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

        # get annotation info
        # instance-seg
        cur_anns_ids = coco.getAnnIds(imgIds=[cur_img_id])
        cur_anns = coco.loadAnns(cur_anns_ids)

        assert cur_anns[0]["image_id"] == cur_img_id

        # save skeleton
        coco_skeleton = np.array(coco.loadCats(cur_anns[0]['category_id'])[0]['skeleton'])
        dataset_meta = parse_pose_metainfo(dict(from_file='/dat03/xuanwenjie/code/animal_pose/mmpose-main/configs/_base_/datasets/ap10k.py'))
        # keypoint_id2name_dict = dataset_meta["keypoint_id2name"]

        # save triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "background": cur_background, 
        }

        # save one json for each image
        save_path = os.path.join(SAVE_ROOT, "{}.json".format(cur_filename_wosuffix))
        # print(save_path)
        with open(save_path, "w") as f:
            json.dump(cur_line, f, indent=4)

        # if idx > -1:
        #     break 

    return 0 



###############################################################################################################
# Main
###############################################################################################################
def main():
    # ap10k
    # prepare_ap10k()

    # humanart
    prepare_humanart()



if __name__ == "__main__":

    main()
