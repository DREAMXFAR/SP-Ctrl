#/bin/bash

################################
# refer to ViTPose

# -----------
CFG_HUGE_PLUS="/dat03/xuanwenjie/code/animal_pose/APTv2-main/configs_vitpose/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/geneval_ViTPose_huge_ap10k_256x192.py"
VIT_HUGE_PLUS_MODEL="/dat03/xuanwenjie/code/animal_pose/APTv2-main/pretrained_models/vitpose+_huge/ap10k.pth"
CONFIG=${CFG_HUGE_PLUS}
CHECKPOINT=${VIT_HUGE_PLUS_MODEL}

NAME="check_controlnet_sd_ap10k_keypoint_ckpt172000"
# NAME="raw_ap10k"

SRC_ROOT="/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/controlnet_gen_eval/${NAME}"
WORK_DIR="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/evaluation/vitpose_eval/work_dir/${NAME}"
POSE_EXCEL_ROOT="/dat03/xuanwenjie/code/release_version/sparse_pose_controlnet/output_dir/evaluation/vitpose_eval/excel_logs"

# run
CUDA_VISIBLE_DEVICES=3 python test.py ${CONFIG} ${CHECKPOINT} --work-dir ${WORK_DIR} --src-root ${SRC_ROOT} --pose-excel-root ${POSE_EXCEL_ROOT}
