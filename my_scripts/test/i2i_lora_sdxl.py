import torch
import cv2
import numpy as np
from PIL import Image
import os

# diffusers から必要なモジュールをインポート
from diffusers import (
    AutoPipelineForImage2Image
)

# ===== 2) Stable Diffusion XL Pipeline の読み込み =====
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# ===== 3) 入力画像の読み込み =====
# ここで任意の画像を読み込みます
input_image_path = "/root/code/diffusers/results/refined_depth_sample/IST02_01_006_C_0001.png"
input_image = Image.open(input_image_path).convert("RGB")

lora_dir = "/root/code/diffusers/output/stable-diffusion-xl-base-1.0"
pipe.load_lora_weights(lora_dir)
lora_scale = 0.9
# ===== 5) 推論（画像生成） =====
prompt = "alleta, full body, front, white background"
result = pipe(
    prompt=prompt,
    image=input_image,         # ControlNet に渡す Canny 画像
    num_inference_steps=30,      # ステップ数
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": lora_scale}, # テキストガイドの強さ
    generator=torch.manual_seed(0),
).images[0]

# ===== 6) 出力画像の保存 =====
os.makedirs("results", exist_ok=True)
result.save("results/i2i.png")