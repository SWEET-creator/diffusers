import torch
import cv2
import numpy as np
from PIL import Image
import os

# diffusers から必要なモジュールをインポート
from diffusers import (
    AutoPipelineForInpainting
)

# ===== 2) Stable Diffusion XL Pipeline の読み込み =====
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16
).to("cuda")

# ===== 3) 入力画像の読み込み =====
# ここで任意の画像を読み込みます
input_image_path = "/root/code/diffusers/no_complete/bidirectional_frame_000.png"
input_image = Image.open(input_image_path).convert("RGB")

# ===== 4) マスク画像の生成 =====
# 画素値が(0,0,0)である部分をマスクにします
np_image = np.array(input_image)
mask = (np_image <= 10).all(axis=-1)
mask_image = Image.fromarray(mask, mode="L")

lora_dir = "/root/code/diffusers/output/lora_alleta_3_pad"
pipe.load_lora_weights(lora_dir)
lora_scale = 0.9
# ===== 5) 推論（画像生成） =====
prompt = "alleta, full body, front, white background"
result = pipe(
    prompt=prompt,
    image=input_image,
    mask_image=mask_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    strength=0.99,
    cross_attention_kwargs={"scale": lora_scale}, # テキストガイドの強さ
    generator=torch.manual_seed(0),
).images[0]

# ===== 6) 出力画像の保存 =====
os.makedirs("results", exist_ok=True)
result.save("results/impaint.png")