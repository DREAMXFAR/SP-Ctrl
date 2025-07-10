import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

import json
import cv2 
from pycocotools.coco import COCO
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 
from tqdm import tqdm 

from controlnet_aux import ZoeDetector


##################################################################################################
# Utils Functions  
##################################################################################################
def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


##################################################################################################
# main 
##################################################################################################

def main():
    ###### AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "train"

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

    # set save path
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/"
    
    depth_save_path = os.path.join(SAVE_ROOT, "zeo_depth")
    if not os.path.exists(depth_save_path):
        os.mkdir(depth_save_path)

    # get all image index info
    ids = list(sorted(coco.imgs.keys()))
    print("==> number of images: {}".format(len(ids)))

    # set detector 
    MODEL_PATH = r"/dat03/xuanwenjie/pretrained_models/ControlNet/Annotators"
    zoe = ZoeDetector.from_pretrained(MODEL_PATH).to("cuda")

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

        cur_img_path = os.path.join(image_path, cur_filename)
        cur_img = Image.open(cur_img_path).convert("RGB")   # .resize((512, 512))

        cur_zoe_depth = zoe(cur_img, output_type="pil")
        cur_zoe_depth = cur_zoe_depth.resize((cur_width, cur_height), Image.BICUBIC)
        # print("cur_width: {}, cur_height: {}".format(cur_width, cur_height))

        save_path = os.path.join(depth_save_path, "{}.png".format(cur_filename_wosuffix))
        cur_zoe_depth.save(save_path)

        # if idx > 0:
        #     break 

    return 0 


def gen_masked_depth():
    
    ### config path
    depth_path = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/zeo_depth/"
    sam_path = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/sam_hq_mask/"

    ### save root 
    save_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/debug/masked_zeo_depth"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    file_list = os.listdir(depth_path)
    num_data = len(file_list)
    pbar = tqdm(file_list)

    for afile in pbar:
        pbar.set_description(f"{num_data}")
        
        depth_image_path = os.path.join(depth_path, afile)
        assert os.path.isfile(depth_image_path)
        sam_image_path = os.path.join(sam_path, afile)
        assert os.path.isfile(sam_image_path)

        depth_image = Image.open(depth_image_path)
        sam_image = Image.open(sam_image_path).convert("L")

        depth_array = np.array(depth_image).astype(np.float32)
        sam_array = np.array(sam_image).astype(np.float32) / 255.0
        sam_array = np.repeat(sam_array[:, :, np.newaxis], 3, axis=2)

        masked_depth_array = depth_array * sam_array
        masked_depth_pil = Image.fromarray(np.uint8(masked_depth_array))

        cur_save_path = os.path.join(save_root, afile)
        masked_depth_pil.save(cur_save_path)

        # break
    return 0


if __name__ == "__main__":

    # main()
    gen_masked_depth()

