source .venv/bin/activate

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export LOCAL_NAME="/root/code/diffusers/checkpoints/Pony_Diffusion_V6_XL.safetensors"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export OUTPUT_DIR="output/lora_alleta_3_pad"
export DATASET_NAME="dataset/lora_alleta_3_pad"

nohup accelerate launch --main_process_port 29501 /root/code/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --pretrained_local_model_name_or_path=$LOCAL_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=5000 \
  --use_8bit_adam \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="alleta" --validation_epochs 5000 \
  --checkpointing_steps=1000 \
  --seed=1337 \
  --output_dir=${OUTPUT_DIR}  > train_log_sdxl.txt 2>&1 &