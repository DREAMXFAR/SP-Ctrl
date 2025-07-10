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
import math 
from scipy.sparse import csr_matrix, save_npz, load_npz 


def main():
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

    ### load annotations 
    coco = COCO(annotation_file=json_path)

    ### class info
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    coco_supercats = {v["name"]: v["supercategory"] for k, v in coco.cats.items()}    

    # set save path
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/"
    
    jsonl_save_path = os.path.join(SAVE_ROOT, "{}_split{}.jsonl".format(split_mode, split_id))
    if os.path.exists(jsonl_save_path):
        os.remove(jsonl_save_path)

    skeleton_save_path = os.path.join(SAVE_ROOT, "{}_split{}_{}_skeleton".format(split_mode, split_id, skeleton_style))
    if not os.path.exists(skeleton_save_path):
        os.mkdir(skeleton_save_path)

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

        cur_img_path = os.path.join(image_path, cur_filename)
        cur_img = np.array(Image.open(cur_img_path))
        cur_canvas = np.zeros_like(cur_img) if black_background else cur_img


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

KEYPOINT_PLACEHOLDER = [
    "<nose>",
    "<left-eye>", 
    "<right-eye>", 
    "<left-ear>", 
    "<right-ear>", 
    "<left-shoulder>", 
    "<right-shoulder>", 
    "<left-elbow>",
    "<right-elbow>", 
    "<left-wrist>",
    "<right-wrist>", 
    "<left-hip>", 
    "<right-hip>", 
    "<left-knee>", 
    "<right-knee>", 
    "<left-ankle>", 
    "<right-ankle>",   
]


def get_humanart_keypoint_prompt():
    ### cfg 
    split_mode = "val"

    check_file = "000000001406.jpg"
    check_category = "dance"

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/HumanArt_triplet/humanart_training.jsonl", 
        "val": r"/dat03/xuanwenjie/datasets/HumanArt_triplet/humanart_validation.jsonl",
    }
    
    image_path = r"/dat03/xuanwenjie/datasets/HumanArt_triplet/images"
    json_path = dataset_split_info[split_mode]

    data = []
    with open(json_path, 'r') as f:
        for aline in f:
            cur_line = json.loads(aline.strip())
            data.append(cur_line)

    for aitem in data:
        cur_filename = aitem["filename"]
        cur_dirname = aitem["dirname"]
        cur_category = aitem['category']

        cur_anns = aitem["anns"]
        cur_caption = aitem["caption"]

        if not cur_filename == check_file:
            continue
        if not cur_category == check_category:
            continue  
        
        kpt_flag = np.zeros(17)
        for ann in cur_anns:
            akpt = ann["keypoints"]
            v_i = akpt[2::3]
            kpt_flag = kpt_flag + np.array(v_i)

        kpt_prompt = ""
        for aid, aflag in enumerate(kpt_flag):
            if aflag > 0: 
                kpt_prompt += "{}, ".format(KEYPOINT_PLACEHOLDER[aid])

        print("kpt_prompt: {}".format(kpt_prompt))
    
    pass


if __name__ == "__main__":
    # run 
    # main()


    get_humanart_keypoint_prompt()