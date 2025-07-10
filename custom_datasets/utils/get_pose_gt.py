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
from scipy.sparse import csr_matrix, save_npz, load_npz 


###############################################################################################
# Functions
###############################################################################################
def write_jsonl(save_path, item):
    # 建立data.jsonl文件，以追加的方式写入数据
    with jsonlines.open(save_path, mode = 'a') as json_writer:
        json_writer.write(item)


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

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

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
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    # draw keypoints
    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x)   # int(x * W)
        y = int(y)   # int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


def draw_bodypose_coco(
        canvas: np.ndarray, 
        keypoints: List[Keypoint], 
        limbSeq, ) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.
    """
    stickwidth = 4

    ac = ((np.random.random((3, ))*0.6+0.4) * 255).tolist()
    print("==> ac = {}".format(ac))
    colors = [ac for i in range(17)]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue
        
        p1 = np.array([keypoint1.x, keypoint1.y])   # * float(W)
        p2 = np.array([keypoint2.x, keypoint2.y])   # * float(H)
        cv2.line(canvas, p1, p2, [int(float(c) * 1.0) for c in color], thickness=3)

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x) 
        y = int(y) 
        
        ### original implementation in plt 
        # plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
        # plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
        
        cv2.circle(canvas, (int(x), int(y)), 8, color, thickness=-1)
        # cv2.circle(canvas, (int(x), int(y)), 10, (0, 0, 0), thickness=1)  # not consider v=1

    return canvas


def draw_bodypose_mmpose(
        canvas: np.ndarray, 
        keypoints: List[Keypoint], 
        limbSeq, ) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.
    """
    # refer to /dat03/xuanwenjie/code/animal_pose/mmpose-main/mmpose/visualization/local_visualizer.py, line 137
    radius = 3
    line_width = 2

    dataset_meta = parse_pose_metainfo(dict(from_file='/dat03/xuanwenjie/code/animal_pose/mmpose-main/configs/_base_/datasets/ap10k.py'))
    colors = dataset_meta["skeleton_link_colors"]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue
        
        p1 = np.array([keypoint1.x, keypoint1.y])   # * float(W)
        p2 = np.array([keypoint2.x, keypoint2.y])   # * float(H)
        cv2.line(canvas, p1, p2, [int(float(c) * 1.0) for c in color], thickness=line_width)

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x) 
        y = int(y) 
        
        cur_color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(canvas, (int(x), int(y)), radius=radius, color=cur_color, thickness=-1)
        # cv2.circle(canvas, (int(x), int(y)), 10, (0, 0, 0), thickness=1)  # not consider v=1

    return canvas


def draw_bodypose_part(
        canvas: np.ndarray, 
        keypoints: List[Keypoint], 
        limbSeq, ) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.
    """
    stickwidth = 4

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for sk_id, (k1_index, k2_index), color in zip(range(canvas.shape[0]), limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        keypoint1_color = colors[k1_index - 1]
        keypoint2_color = colors[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x])   # * float(W)
        X = np.array([keypoint1.y, keypoint2.y])   # * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas[sk_id, :], polygon, [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip((keypoint1, keypoint2), (keypoint1_color, keypoint2_color)):
            x, y = keypoint.x, keypoint.y
            x = int(x)   # int(x * W)
            y = int(y)   # int(y * H)
            cv2.circle(canvas[sk_id, :], (int(x), int(y)), 4, color, thickness=-1)       

    return canvas



###################################################################################################
# Utils
###################################################################################################
def check_dataset_info():
    ### AP-10k 
    split_id = 1
    train_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id)
    val_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id)
    test_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id)

    coco_train = COCO(annotation_file=train_split_json)
    coco_val = COCO(annotation_file=val_split_json)
    coco_test = COCO(annotation_file=test_split_json)

    ### 1. check all splits are summed to the whole dataset
    # get all image index info
    train_ids = list(sorted(coco_train.imgs.keys()))
    val_ids = list(sorted(coco_val.imgs.keys()))
    test_ids = list(sorted(coco_test.imgs.keys()))

    print("Split [{}] ==>  number of images: \n\ttrain ={}, val={}, test={}".format(split_id, len(train_ids), len(val_ids), len(test_ids)))
    print("\tsum = {}".format(len(train_ids) + len(val_ids) + len(test_ids)))


def main():
    ###### AP-10K 
    ### cfg 
    split_id = 1
    split_mode = "train"
    black_background = True
    skeleton_style = "openpose"  # "mmpose" 

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
 
        ### draw pose 
        for ann in cur_anns:
            # turn skeleton into zero-based index
            sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])  # -1
            kp = np.array(ann['keypoints'])

            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]

            keypoint_list = []
            for kpx, kpy, kpv in zip(x,y,v):
                if kpv == 0:
                    keypoint_list.append(None)
                else:
                    keypoint_list.append(Keypoint(x=kpx, y=kpy, score=1.0, id=-1))

            if skeleton_style == "openpose":
                # openpose style 
                cur_canvas = draw_bodypose(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            elif skeleton_style == "coco":
                # coco style 
                cur_canvas = draw_bodypose_coco(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            elif skeleton_style == "mmpose":
                # mmpose style, refer to test_visualizer.py
                cur_canvas = draw_bodypose_mmpose(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            else:
                raise NotImplementedError

        sks_save_path = os.path.join(skeleton_save_path, cur_filename_wosuffix + ".png")
        cur_canvas_pil = Image.fromarray(np.uint8(cur_canvas))
        cur_canvas_pil.save(sks_save_path)

        # save triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "background": cur_background, 
        }

        write_jsonl(jsonl_save_path, cur_line)

        # if idx > 10:
        #     break 

    print("==> background_catid: {}".format(background_cats))

    return 0 


def generate_ap10k_category(debug=False):
    ### AP-10k 
    split_id = 1
    train_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-train-split{}.json".format(split_id)
    val_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split{}.json".format(split_id)
    test_split_json = r"/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-test-split{}.json".format(split_id)

    coco_train = COCO(annotation_file=train_split_json)
    coco_val = COCO(annotation_file=val_split_json)
    coco_test = COCO(annotation_file=test_split_json)

    ### basic info
    dataset_len = len(coco_train.cats)
    count = 0

    # set save path
    DATASET_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/"
    save_cat_path = os.path.join(DATASET_ROOT, "catgories.jsonl")
    # if os.path.exists(save_cat_path):
    #     os.remove(save_cat_path)

    # for loop
    rewrite_json ={}
    
    pbar = tqdm(coco_train.cats.keys())
    for idx in pbar:
        count += 1
        pbar.set_description("[category]")

        cur_train_cat = coco_train.cats[idx]
        cur_val_cat = coco_val.cats[idx]

        ### get train 
        cur_train_id = cur_train_cat["id"]
        cur_train_name = cur_train_cat["name"]
        cur_train_supercategory = cur_train_cat["supercategory"]
        cur_train_keypoints = cur_train_cat["keypoints"]
        cur_train_skeleton = cur_train_cat["skeleton"]

        ### get val
        cur_val_id = cur_val_cat["id"]
        assert cur_val_id == cur_train_id
        cur_val_name = cur_val_cat["name"]
        assert cur_val_name == cur_train_name
        cur_val_supercategory = cur_val_cat["supercategory"]
        assert cur_val_supercategory == cur_train_supercategory
        cur_val_keypoints = cur_val_cat["keypoints"]
        assert cur_val_keypoints == cur_train_keypoints
        cur_val_skeleton = cur_val_cat["skeleton"]
        assert cur_val_skeleton == cur_train_skeleton

        ### not same 
        # assert cur_val_image_count == cur_train_image_count 
        # assert cur_val_instance_count == cur_train_instance_count

        ### collect triplet pair
        rewrite_json[idx] = {
            "id": cur_train_id, 
            "name": cur_train_name, 
            "supercategory": cur_train_supercategory, 
            "keypoints": cur_train_keypoints, 
            "skeleton": cur_train_skeleton, 
        }

        if debug and count > 2:
            break
    
    with open(save_cat_path, 'w') as f:
        json.dump(rewrite_json, f, indent=4)

    print("==> Finish!")


def get_human_pose_gt():
    from mmengine.structures import InstanceData
    from mmpose.structures import PoseDataSample
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.visualization import PoseLocalVisualizer

    split = "val"
    black_background = True
    skeleton_style = "mmpose"  # "openpose"

    json_path = r"/dat03/xuanwenjie/datasets/COCO/annotations/person_keypoints_{}2017.json".format(split)
    image_path = r"/dat03/xuanwenjie/datasets/COCO/val2017"
    coco = COCO(annotation_file=json_path)

    caption_json_path = r"/dat03/xuanwenjie/datasets/COCO/annotations/captions_{}2017.json".format(split)
    caption_coco = COCO(annotation_file=caption_json_path)
    
    # get all image index info
    ids = list(sorted(coco.imgs.keys()))
    print("==> number of images: {}".format(len(ids)))

    # save_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/coco_keypoints/{}".format(split)
    # if not os.path.exists(save_root):
    #     os.mkdir(save_root)
    
    # json_save_path = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/coco_keypoints/caption.jsonl".format(split)
    # if os.path.exists(json_save_path):
    #     os.remove(json_save_path)


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

        # get annotation info
        # instance-seg
        cur_anns_ids = coco.getAnnIds(imgIds=[cur_img_id])
        cur_anns = coco.loadAnns(cur_anns_ids)

        cur_caption_ids = caption_coco.getAnnIds(imgIds=[cur_img_id])
        cur_captions = caption_coco.loadAnns(cur_caption_ids)

        if len(cur_anns) == 0:
            continue

        assert cur_anns[0]["image_id"] == cur_img_id

        # save skeleton
        coco_skeleton = np.array(coco.loadCats(cur_anns[0]['category_id'])[0]['skeleton'])
        dataset_meta = parse_pose_metainfo(dict(from_file='/dat03/xuanwenjie/code/animal_pose/mmpose-main/configs/_base_/datasets/coco.py'))

        cur_img_path = os.path.join(image_path, cur_filename)
        cur_img = np.array(Image.open(cur_img_path))
        cur_canvas = np.zeros_like(cur_img) if black_background else cur_img
 
        flag = 0
        
        if len(cur_anns) > 2:  # filter multiple instances 
            continue  

        ### draw pose 
        for ann in cur_anns:

            if ann["num_keypoints"] == 0:
                flag = 1
                continue

            if ann["num_keypoints"] < 15:
                flag = 1
                continue

            # turn skeleton into zero-based index
            sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])  # -1
            kp = np.array(ann['keypoints'])

            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]

            keypoint_list = []
            for kpx, kpy, kpv in zip(x,y,v):
                if kpv == 0:
                    keypoint_list.append(None)
                else:
                    keypoint_list.append(Keypoint(x=kpx, y=kpy, score=1.0, id=-1))

            if skeleton_style == "openpose":
                # openpose style 
                cur_canvas = draw_bodypose(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            elif skeleton_style == "coco":
                # coco style 
                cur_canvas = draw_bodypose_coco(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            elif skeleton_style == "mmpose":
                # mmpose style, refer to test_visualizer.py
                cur_canvas = draw_bodypose_mmpose(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks)
            else:
                raise NotImplementedError

        if flag == 1:
            continue
        
        # save triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "caption": cur_captions, 
        }

        # write_jsonl(json_save_path, cur_line)

        # if idx > 3:
        #     break 

    return 0 


def get_animal_pose_keypoint():
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
    SAVE_ROOT = r"/dat03/xuanwenjie/datasets/AP-10K_triplet/only_keypoint"

    skeleton_save_path = os.path.join(SAVE_ROOT, "{}_split{}_{}_keypoint".format(split_mode, split_id, skeleton_style))
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
 
        ### draw pose 
        for ann in cur_anns:
            # turn skeleton into zero-based index
            sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])  # -1
            kp = np.array(ann['keypoints'])

            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]

            keypoint_list = []
            for kpx, kpy, kpv in zip(x,y,v):
                if kpv == 0:
                    keypoint_list.append(None)
                else:
                    keypoint_list.append(Keypoint(x=kpx, y=kpy, score=1.0, id=-1))

            if skeleton_style == "openpose":
                # openpose style 
                cur_canvas = draw_bodypose(canvas=cur_canvas, keypoints=keypoint_list, limbSeq=sks, only_keypoints=True)
            else:
                raise NotImplementedError

        sks_save_path = os.path.join(skeleton_save_path, cur_filename_wosuffix + ".png")
        cur_canvas_pil = Image.fromarray(np.uint8(cur_canvas))
        cur_canvas_pil.save(sks_save_path)

        # save triplet pair
        cur_line = {
            "filename": cur_filename, 
            "image_id": cur_img_id, 
            "height": cur_height,
            "width": cur_width, 
            "anns": cur_anns,
            "background": cur_background, 
        }

        # break

    print("==> background_catid: {}".format(background_cats))

    return 0 


if __name__ == "__main__":

    ### 1. save animal skeleton and json for dataset  
    # main()

    ### 2. save category json
    # generate_ap10k_category()

    ### 3. get human pose gt 
    get_human_pose_gt()

    ### 4. vis only pose keypoint (wo-skeleton)
    # get_animal_pose_keypoint()

    print(":) Congratulations!")