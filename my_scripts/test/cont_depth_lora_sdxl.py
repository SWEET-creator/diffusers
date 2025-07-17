import torch
import cv2
import numpy as np
from PIL import Image
import os

# diffusers から必要なモジュールをインポート
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)

# ===== 1) ControlNet の読み込み =====
# Stable Diffusion XL 用の Canny ControlNet
# （例: diffusers公式の "diffusers/controlnet-canny-sdxl-1.0" などを使用）
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
).to("cuda")

# ===== 2) Stable Diffusion XL Pipeline の読み込み =====
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# ===== 3) 入力画像の読み込み =====
# ここで任意の画像を読み込みます
user_input = "/root/code/diffusers/dataset/tooncrafter_IST02_01_006_C_0001"

# ディレクトリか画像かを判定
if os.path.isdir(user_input):
    input_image_list = [os.path.join(user_input, x) for x in os.listdir(user_input)]
else:
    input_image_list = [user_input]


pipe.load_lora_weights("/root/code/diffusers/output/lora_alleta_3_pad/checkpoint-1000", adapter_name="basic")
# pipe.load_lora_weights("/root/code/diffusers/output/lora_alleta_face", adapter_name="face")
# pipe.set_adapters(["basic", "face"], adapter_weights=[0.9, 0.9])
lora_scale = 0.9
prompt = "alleta, white background"

ip_adapter_iamge_path = None #"/root/code/diffusers/dataset/lora_alleta_face/IST02_01_006_C_0001.png"
ip_adapter_image = None
if ip_adapter_iamge_path != None:
    ip_adapter_image = Image.open(ip_adapter_iamge_path).convert("RGB")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.9)
    
# for文で処理
for input_image_path in input_image_list:
    input_image = Image.open(input_image_path).convert("RGB")
    input_image_name = os.path.splitext(os.path.basename(input_image_path))[0]
    
    if input_image.size[0] < 1024 or input_image.size[1] < 1024:
        # keep aspect ratio and resize
        min_size = min(input_image.size)
        input_image = input_image.resize((int(input_image.size[0] * 1024 / min_size), int(input_image.size[1] * 1024 / min_size)))

    result = pipe(
        prompt=prompt,
        image=input_image,         # ControlNet に渡す Canny 画像
        ip_adapter_image=ip_adapter_image,
        num_inference_steps=30,      # ステップ数
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": lora_scale}, # テキストガイドの強さ
        generator=torch.manual_seed(0),
    ).images[0]

    # ===== 6) 出力画像の保存 =====
    if os.path.isdir(user_input):
        dir_name = os.path.basename(user_input)
        os.makedirs(f"results/{dir_name}", exist_ok=True)
        result.save(f"results/{dir_name}/{input_image_name}.png")
    else:
        os.makedirs("results", exist_ok=True)
        result.save(f"results/{input_image_name}.png")