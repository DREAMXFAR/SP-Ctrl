import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # solve RuntimeError: main thread is not in main loop
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# from segment_anything.utils.amg import coco_encode_rle, mask_to_rle_pytorch
import json
from tqdm import tqdm 
import time
from segment_anything_hq import sam_model_registry, SamPredictor
from matplotlib import font_manager, rcParams
import codecs
 
from pycocotools.coco import COCO
from PIL import Image
import random 


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x+w, y+h])


def gray_to_rgb(image_gray, fg_color=(255, 0, 0), bg_color=(0, 0, 0)):
    # create a zero array with 3 channels
    if isinstance(image_gray, torch.Tensor):
        image_gray = image_gray.cpu().squeeze().numpy() 

    height, width = image_gray.shape
    image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # copy values
    # fg_color, bg_color all in RGB
    image_rgb[:, :, 0][image_gray > 0] = fg_color[0]
    image_rgb[:, :, 1][image_gray > 0] = fg_color[1]
    image_rgb[:, :, 2][image_gray > 0] = fg_color[2]

    image_rgb[:, :, 0][image_gray == 0] = bg_color[0]
    image_rgb[:, :, 1][image_gray == 0] = bg_color[1]
    image_rgb[:, :, 2][image_gray == 0] = bg_color[2]

    return image_rgb


def main():
    ### set SAM 
    sam_type = "sam_hq"
    
    if sam_type == "sam":  # conda_env: sam
        sam_checkpoint = "/dat02/checkpoints/sam/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    elif sam_type == "sam_hq": # conda_env: tmp-diffusers
        sam_checkpoint = "/dat03/xuanwenjie/pretrained_models/SAM/sam_hq_vit_h.pth"
        model_type = 'vit_h'
        
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    ### load dataset 
    # AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "test"  # "train"  # "val"
    black_background = True

    # basic info 
    dataset_split_info = {
        "train": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id), 
        "val": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id),
        "test": r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id), 
    }
    
    image_path = r"/dat03/xuanwenjie/datasets/AP-10K/data/"
    json_path = dataset_split_info[split_mode]

    # save_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/check_results/sam_imgs"
    save_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/check_results/sam_hq_ap10k_masks"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    ### load annotations 
    coco = COCO(annotation_file=json_path)

    ### class info
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    coco_supercats = {v["name"]: v["supercategory"] for k, v in coco.cats.items()}    

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
        
        cur_img_path = os.path.join(image_path, cur_filename)
        cur_img = np.array(Image.open(cur_img_path))

        # visualization
        cur_canvas = np.zeros_like(cur_img) if black_background else cur_img
        cur_mask = np.zeros((cur_img.shape[0], cur_img.shape[1]))

        # get annotation info
        # instance-seg
        cur_anns_ids = coco.getAnnIds(imgIds=[cur_img_id])
        cur_anns = coco.loadAnns(cur_anns_ids)

        assert cur_anns[0]["image_id"] == cur_img_id
        for ann in cur_anns: 
            bbox = ann["bbox"]
            bbox = xywh2xyxy(bbox)
            cat_id = ann["category_id"]

            # set predictor
            predictor = SamPredictor(sam)
            predictor.set_image(cur_img)

            prompt_mode = "mask"
            if prompt_mode == "point":
                # keypoints prompt
                kp = np.array(ann['keypoints']).reshape(17, 3)[:, :2]
                kp_wozeros = []
                for kp_i in kp:
                    if np.sum(kp_i) != 0:
                        kp_wozeros.append(kp_i)
                kp = np.array(kp_wozeros)
                kp = torch.from_numpy(kp).unsqueeze(0).cuda()
                kp_labels = torch.ones(kp.shape[1], device='cuda').unsqueeze(0)

                masks, qualities, resolutions = predictor.predict_torch(
                    point_coords=kp,
                    point_labels=kp_labels,
                    boxes=None,
                    multimask_output=False,
                )
            elif prompt_mode == "mask":
                # for bbox prompt
                bbox = torch.tensor(bbox, device=device)
                transformed_boxes = predictor.transform.apply_boxes_torch(bbox, (cur_width, cur_height))
                
                masks, qualities, resolutions = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                raise Exception("!!! Wrong prompt mode !!!")

            # for mask, conf in zip(masks, qualities.data):
            #     ### for visualization
            #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #     cv2.rectangle(cur_canvas, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=color, thickness=3)
            #     cv2.putText(cur_canvas, str(float(conf)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color)
            #     mask_vis = gray_to_rgb(mask, fg_color=color)

            #     # merge two images 
            #     alpha = 0.3
            #     cur_canvas = cv2.addWeighted(cur_canvas, 1.0, mask_vis, (1-alpha), gamma=0)

            #     # ## not used: mask2rle, pycocotools.decode -> mask
            #     # mask_rle = coco_encode_rle(mask_to_rle_pytorch(mask)[0])
            #     # mask_rle["conf"] = float(conf)
            # save_path = os.path.join(save_root, "{}.png".format(cur_filename_wosuffix))
            # cur_canvas_pil = Image.fromarray(np.uint8(cur_canvas))
            # cur_canvas_pil.save(save_path)
            
            for mask, conf in zip(masks, qualities.data):
                cur_mask = cur_mask + mask.cpu().squeeze().numpy()
            
            save_path = os.path.join(save_root, "{}.png".format(cur_filename_wosuffix))
            cur_canvas_pil = Image.fromarray(np.uint8(255 * (cur_mask > 0)))
            cur_canvas_pil.save(save_path)

        # debug use
        # if idx > 2:
        #     break


if __name__ == "__main__":
    main()
