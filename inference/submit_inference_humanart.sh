#/bin/bash

#################################################################################################################
### Inference Task
# cfg
COND_MODE="ap_spatext"

# CONTROL_IMAGE="/dat03/xuanwenjie/datasets/HumanArt_triplet/humanart_kptindex/real_human/dance/000000000274.png"
# RAW_IMAGE="/dat03/xuanwenjie/datasets/HumanArt/images/real_human/dance/000000000274.jpg"
# PROMPT="dance, two dancers on the stage. <nose>, <left-eye>, <right-eye>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>"

CONTROL_IMAGE="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/data/demo/cartoon_000000000053/pose_kptindex.png"
RAW_IMAGE="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/data/demo/cartoon_000000000053/raw_image.jpg"
PROMPT="cartoon, a girl in red. <nose>, <left-eye>, <right-eye>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>"


# CONTROL_IMAGE="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/data/demo/garage_kits_000000000768/pose_kptindex.png"
# RAW_IMAGE="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/data/demo/garage_kits_000000000768/raw_image.jpg"
# PROMPT="garage kits, a cute girl with pink hear. <nose>, <left-eye>, <right-eye>, <right-ear>, <left-shoulder>, <right-shoulder>, <left-elbow>, <right-elbow>, <left-wrist>, <right-wrist>, <left-hip>, <right-hip>, <left-knee>, <right-knee>, <left-ankle>, <right-ankle>"

CONTROLNET_PATH="/dat04/xuanwenjie/supercom/tmp/transfer_dirs/sc_controlnet_sd_humanart_keypoint_crossattnloss_partition_kpttoken_mseloss_timestep_250_500_downblocks_2_detachq_spatext_kptrand_sksone/checkpoint-528000/controlnet"
OUTPUT_ROOT="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/demo"


# script
CUDA_VISIBLE_DEVICES=3 python inference/gen_keypoint_w_controlnet_humanart.py \
    --control_image_path=${CONTROL_IMAGE} \
    --raw_image_path=${RAW_IMAGE} \
    --prompt="${PROMPT}" \
    --controlnet_path=${CONTROLNET_PATH} \
    --output_root=${OUTPUT_ROOT} \
    --cond_mode=${COND_MODE} \
    --enable_learnable_token


# hint info 
echo "# ------------------------------------------------------------------------"
echo "# "${CONTROLNET_PATH}
echo "# "${OUTPUT_ROOT}
echo "# ------------------------------------------------------------------------"