import os 
import numpy as np
import json
import random 
from tqdm import tqdm
import re

from model_config import *


def get_pose_token(image_name, in_format=True):
    ### prepare
    image_name_wosuffix = os.path.splitext(os.path.basename(image_name))[0]

    # get category json 
    category_json_path = os.path.join(DATASET_ROOT["ap10k"], r"catgories.jsonl")
    with open(category_json_path, 'r') as f:
        category_json = json.load(f)

    ### load data from jsonl 
    jsonl_path = TEST_JSON["ap10k-val-split-1"]
    data = []
    with open(jsonl_path, 'r') as f:
        json_data = f.readlines()
        for line in json_data:
            data.append(json.loads(line))
    num_data = len(data)

    ### loop 
    pbar = tqdm(range(num_data))
    for idx in pbar:
        aitem = data[idx]
        
        cur_filename = aitem["filename"].split(".")[0]
        if not cur_filename == image_name_wosuffix:
            continue
        
        background = BACKGROUND_CATEGORY[str(aitem["background"])]
        instance_cats = []
        kpt_vis = np.zeros(NUM_KEYPOINTS)
        
        for ann in aitem["anns"]:
            ann_name = category_json[str(ann['category_id'])]["name"]
            cur_kpt_vis = ann["keypoints"][2::3]
            kpt_vis = kpt_vis + np.array(ann["keypoints"][2::3])

            if ann_name in instance_cats:
                continue  # filter duplicated category name
            else:
                instance_cats.append(ann_name)
        kpt_vis = kpt_vis > 0
        
        ### get image caption as prompt
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
        prompt = random.choice(caption_templates)

        # preprocess text
        kpt_vis_list = []
        for avis, akptname in zip(kpt_vis, KEYPOINT_NAME):
            if in_format:
                akptname = "<" + "-".join(akptname.split(" ")) + ">"
            if avis:
                kpt_vis_list.append(akptname) 
        keypoints_prompt = " <" + ", ".join(kpt_vis_list) + ">"
        prompt = re.sub(r"<KPT>", keypoints_prompt, prompt)

        print("#" * 100)
        print("#         image_id: {}".format(cur_filename))
        print("#    inst_category: {}".format(" & ".join(instance_cats)))
        print("#    num_keypoints: {}".format(len(kpt_vis_list)))
        print("# keypoints_prompt: {}".format(keypoints_prompt))
        print("# candidate_prompt: {}".format(prompt))
        print("#" * 100)


if __name__ == "__main__":
    # main
    image_name = "000000020512.jpg"
    get_pose_token(image_name=image_name)