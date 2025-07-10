#/bin/bash

#################################################################################################################
### Inference Task
# cfg
# COND_MODE="ap"
COND_MODE="ap_samhqmask"

CONTROL_IMAGE="./data/demo/ap10k_000000000222/mask.png"
RAW_IMAGE="./data/demo/ap10k_000000000222/raw_image.jpg"
PROMPT="a tiger in the wild"

CONTROLNET_PATH="/path/to/controlnet_ap10k_mask"
OUTPUT_ROOT="./output_dir/demo"


# script
CUDA_VISIBLE_DEVICES=0 python inference/gen_keypoint_w_controlnet.py \
    --control_image_path=${CONTROL_IMAGE} \
    --raw_image_path=${RAW_IMAGE} \
    --prompt="${PROMPT}" \
    --controlnet_path=${CONTROLNET_PATH} \
    --output_root=${OUTPUT_ROOT} \
    --cond_mode=${COND_MODE}

# hint info 
echo "# ------------------------------------------------------------------------"
echo "# "${CONTROLNET_PATH}
echo "# "${OUTPUT_ROOT}
echo "# ------------------------------------------------------------------------"


#################################################################################################################
### Inference Task
# cfg
COND_MODE="ap_spatext"

CONTROL_IMAGE="./data/demo/ap10k_000000000222/pose_kptindex.png"
RAW_IMAGE="./data/demo/ap10k_000000000222/raw_image.jpg"
PROMPT="a tiger in the wild"

CONTROLNET_PATH="/path/to/spctrl_ap10k"
OUTPUT_ROOT="./output_dir/demo"


# script
CUDA_VISIBLE_DEVICES=0 python inference/gen_keypoint_w_controlnet.py \
    --control_image_path=${CONTROL_IMAGE} \
    --raw_image_path=${RAW_IMAGE} \
    --prompt="${PROMPT}" \
    --controlnet_path=${CONTROLNET_PATH} \
    --output_root=${OUTPUT_ROOT} \
    --cond_mode=${COND_MODE} \
    --spatext_skeleton_type="one" \
    --enable_learnable_token

# hint info 
echo "# ------------------------------------------------------------------------"
echo "# "${CONTROLNET_PATH}
echo "# "${OUTPUT_ROOT}
echo "# ------------------------------------------------------------------------"
