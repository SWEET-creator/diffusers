source .venv/bin/activate

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="output/stable-diffusion-v1-5-geo-train"
export DATASET_NAME="/mnt/data/geo_dataset"
export ACCELERATE_CONFIG_FILE="/mnt/data/models/accelerate/default_config.yaml"

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=1


nohup accelerate launch /root/code/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --checkpointing_steps=2000 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --caption_column caption \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR > pre_train_log.txt 2>&1 &