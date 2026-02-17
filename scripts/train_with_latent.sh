export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== Configuration ==========
MODEL_PATH="./pretrained/CogVideoX-5b-I2V"
video_root_dir="./data/Wan_sample_720p_outputs"  #
output_dir=./out/i2v_context_anchor_joint_attn_final_stage2
# ====================================================
cd training

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file accelerate_config_machine.yaml --multi_gpu --main_process_port 29502 \
  train_controlnet_i2v_use_latent.py \
  --tracker_name "cogvideox-controlnet" \
  --gradient_checkpointing \
  --enable_tiling \
  --enable_slicing \
  --seed 3407 \
  --mixed_precision bf16 \
  --output_dir $output_dir \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_root_dir $video_root_dir \
  --hflip_p 0.0 \
  --controlnet_transformer_num_layers 16 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 1.0 \
  --train_batch_size 1 \
  --dataloader_num_workers 8 \
  --num_train_epochs 8 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 50 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --enable_time_sampling \
  --time_sampling_type truncated_uniform \
  --controlnet_guidance_start 0.0 \
  --controlnet_guidance_end 1.0 \
  --ema_decay 0.995 \
  --motion_sub_loss \
  --motion_sub_loss_ratio 0.2 \
  --controlnet_transformer_num_attn_heads 16 \
  --controlnet_transformer_attention_head_dim 64 \
  --controlnet_transformer_out_proj_dim_factor 64 \
  --controlnet_transformer_out_proj_dim_zero_init \
  --use_camera_condition

