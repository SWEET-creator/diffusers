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
    "diffusers/controlnet-canny-sdxl-1.0",
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
input_image_path = "/root/code/diffusers/dataset/lora_alleta_face/IST02_01_006_C_0001.png"
input_image = Image.open(input_image_path).convert("RGB")

# ===== 4) Canny エッジ画像の生成 =====
# OpenCV を使って Canny 変換します
np_image = np.array(input_image)
canny = cv2.Canny(np_image, 100, 200)  # しきい値は調整可能
# 白黒のエッジ画像を3チャンネル(RGB)にして PIL.Image に変換
canny_3ch = np.stack([canny, canny, canny], axis=-1)
control_image = Image.fromarray(canny_3ch, mode="RGB")

lora_dir = "/root/code/diffusers/output/lora_alleta_3_pad/checkpoint-1000"
pipe.load_lora_weights(lora_dir)
lora_scale = 0.9
# ===== 5) 推論（画像生成） =====
prompt = "alleta, white background"
result = pipe(
    prompt=prompt,
    image=control_image,         # ControlNet に渡す Canny 画像
    num_inference_steps=30,      # ステップ数
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": lora_scale}, # テキストガイドの強さ
    generator=torch.manual_seed(0),
).images[0]

# ===== 6) 出力画像の保存 =====
os.makedirs("results", exist_ok=True)
result.save("results/result_canny_control.png")

# 黒背景と白線のCannyエッジ画像を透明背景・黒線に変換
canny_transparent = np.zeros((canny.shape[0], canny.shape[1], 4), dtype=np.uint8)
canny_transparent[..., 3] = canny  # アルファチャンネルにCannyエッジを設定（線部分を透明以外に）
canny_transparent[..., 0:3] = 0  # RGB部分を黒に設定
canny_image_pil = Image.fromarray(canny_transparent, mode="RGBA")

canny_image_pil.save("results/edge_image.png")

# ===== 7) 元画像に重ねる処理 =====
# 入力画像をRGBAモードに変換
input_image_rgba = input_image.convert("RGBA")
# Cannyエッジ画像を元画像に重ねる

# モードをチェックして適切に変換
if result.mode != "RGBA":
    result = result.convert("RGBA")
if canny_image_pil.mode != "RGBA":
    canny_image_pil = canny_image_pil.convert("RGBA")

# risize
result = result.resize(input_image.size)
canny_image_pil = canny_image_pil.resize(input_image.size)

combined_image = Image.alpha_composite(result, canny_image_pil)

# ===== 8) 出力画像の保存 =====
os.makedirs("results", exist_ok=True)
result.save("results/result_canny_control.png")
combined_image.save("results/combined_image.png")
print("画像を生成・保存しました:")
print("  - results/result_canny_control.png")
print("  - results/combined_image.png")