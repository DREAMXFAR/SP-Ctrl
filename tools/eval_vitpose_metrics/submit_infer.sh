#/bin/bash

# refer to: https://github.com/open-mmlab/mmpose/blob/v0.22.0/demo/docs/2d_animal_demo.md

export CUDA_VISIBLE_DEVICES=2


CFG_BASE_PLUS="/dat03/xuanwenjie/code/animal_pose/APTv2-main/configs_vitpose/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/geneval_ViTPose_base_ap10k_256x192.py"
VIT_BASE_PLUS_MODEL="/dat03/xuanwenjie/code/animal_pose/APTv2-main/pretrained_models/vitpose+_base/ap10k.pth"

CONFIG=${CFG_BASE_PLUS}
CHECKPOINT=${VIT_BASE_PLUS_MODEL}

IMG_ROOT="/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/evaluation/test_data_cache"
# IMG_ROOT="/dat03/xuanwenjie/datasets/AP-10K_triplet/data"
JSON_PATH="/dat03/xuanwenjie/datasets/AP-10K/annotations/ap10k-val-split1.json"

python top_down_img_demo.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --img-root ${IMG_ROOT} \
    --json-file ${JSON_PATH} \
    --out-img-root /dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/vitpose_vis_results/controlnet_ap10k_val


### offical demo
# python top_down_img_demo.py \
#     /dat03/xuanwenjie/code/animal_pose/APTv2-main/configs_vitpose/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res50_macaque_256x192.py \
#     /dat03/xuanwenjie/code/animal_pose/APTv2-main/pretrained_models/res50_macaque_256x192-98f1dd3a_20210407.pth \
#     --img-root /dat03/xuanwenjie/code/animal_pose/APTv2-main/tests/data/macaque/ \
#     --json-file /dat03/xuanwenjie/code/animal_pose/APTv2-main/tests/data/macaque/test_macaque.json \
#     --out-img-root /dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/vis_results