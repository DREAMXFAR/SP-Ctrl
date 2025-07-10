#!/bin/bash
export HF_HOME="/dat03/xuanwenjie/cache_dir/hf_cache" 
export MODEL_DIR="/dat03/xuanwenjie/pretrained_models/StableDiffusion/stable-diffusion-v1-5"
export OUTPUT_DIR="/dat03/xuanwenjie/code/controlnet/output_dir/debug/sh_train"

export CUDA_VISIBLE_DEVICES=3

DATASET_NAME="ap10k"
CONDITIONING_MODE="spatext"

TRACKER_PROJECT_NAME="sp-ctrl_ap10k_demo"

VALIDATION_STEPS=5

### for spatext-image test
VAL_ROOT="/dat03/xuanwenjie/datasets/AP-10K_triplet/pose_kptindex/"
VALIDATION_IMAGE_1=${VAL_ROOT}'000000035817.png'
VALIDATION_PROMPT_1='A photo of cat in the home'
VALIDATION_IMAGE_2=${VAL_ROOT}'000000001996.png'
VALIDATION_PROMPT_2='A photo of buffalo on the grass'
VALIDATION_IMAGE_3=${VAL_ROOT}'000000041035.png'
VALIDATION_PROMPT_3='A photo of tiger in the zoo'

accelerate config default 
accelerate launch --num_processes=1 --mixed_precision=fp16 train_controlnet_mypose.py\
  --pretrained_model_name_or_path=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --conditioning_mode=$CONDITIONING_MODE \
  --spatext_cond_channels=16 \
  --resolution=512 \
  --learning_rate=1e-5 \
  --dataloader_num_workers=0 \
  --checkpointing_steps=10 \
  --validation_steps=$VALIDATION_STEPS \
	--validation_image $VALIDATION_IMAGE_1 $VALIDATION_IMAGE_3 $VALIDATION_IMAGE_2 \
  --validation_prompt "${VALIDATION_PROMPT_1}" "${VALIDATION_PROMPT_3}" "${VALIDATION_PROMPT_2}" \
  --train_batch_size=2 \
	--tracker_project_name=$TRACKER_PROJECT_NAME \
	--num_train_epochs=1 \
  --max_train_steps=20 \
  --proportion_empty_prompts=0.5 

### recommended setti
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes=1 --mixed_precision=fp16 --main_process_port=29072\
#   train_controlnet_mypose.py \
#   --pretrained_model_name_or_path=$MODEL_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --dataset_name=$DATASET_NAME \
#   --conditioning_mode=$CONDITIONING_MODE \
#   --add_keypoint_prompt=0.8 \
#   --add_cross_attn_loss \
#   --heatmap_type="partition" \
#   --attn_loss_type="mse" \
#   --attn_loss_select_layer_type="down_blocks.2" \
#   --attn_loss_start_timestep=250 \
#   --attn_loss_end_timestep=500 \
#   --enable_learnable_kpttoken \
#   --attn_loss_detach_query \
#   --text_encoder_lr=0.5 \
#   --text_encoder_lr_scheduler=0.5 \
#   --attn_loss_weight=0.1 \
#   --spatext_cond_channels=3 \
#   --spatext_skeleton_type="one" \
#   --resolution=512 \
#   --learning_rate=1e-5 \
#   --dataloader_num_workers=8 \
#   --checkpointing_steps=8000 \
#   --validation_steps=800 \
# 	--validation_image $VALIDATION_IMAGE_1 $VALIDATION_IMAGE_2 $VALIDATION_IMAGE_3 \
#   --validation_prompt "${VALIDATION_PROMPT_1}" "${VALIDATION_PROMPT_2}" "${VALIDATION_PROMPT_3}" \
#   --train_batch_size=20 \
# 	--tracker_project_name=$TRACKER_PROJECT_NAME \
# 	--num_train_epochs=800 \
#   --proportion_empty_prompts=0.5 \
#   --gradient_accumulation_steps 1 \
#   --report_to="tensorboard" \
#   --logging_dir="/scratch/jhliu4/xwj/code/controlnet/log/tensorboard_logs/" 


