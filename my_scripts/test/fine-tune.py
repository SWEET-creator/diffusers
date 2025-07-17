import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_dir = "/root/code/diffusers/output/stable-diffusion-v1-5-geo-train"
output_dir = "/root/code/diffusers/output/results/"
os.makedirs(output_dir, exist_ok=True)

# LoRA のスケール等、必要なら設定
lora_scale = 1.0

# ベースモデルを読み込む
pipe_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    pipe_id,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda")

# 生成用のプロンプト
prompt = "star (motion: translation) at (211, 222), size: 64, angle: 150, color: (118, 109, 221); star (motion: all) at (346, 218), size: 65, angle: 132, color: (185, 248, 32); bezier (motion: scaling) at (219, 201), size: 64, angle: 8, color: (221, 48, 192); bezier (motion: translation) at (356, 197), size: 64, angle: 42, color: (116, 24, 239); bezier (motion: all) at (202, 171), size: 61, angle: 59, color: (227, 211, 250); circle (motion: scaling) at (165, 325), size: 63, angle: 263, color: (125, 89, 146); circle (motion: scaling) at (236, 368), size: 64, angle: 312, color: (241, 61, 153); polygon (motion: all) at (159, 157), size: 65, angle: 51, color: (40, 229, 164); polygon (motion: scaling) at (248, 284), size: 67, angle: 19, color: (129, 123, 79); ellipse (motion: scaling) at (319, 216), size: 66, angle: 193, color: (167, 18, 215); bezier (motion: translation) at (278, 296), size: 64, angle: 193, color: (111, 3, 172); star (motion: rotation) at (304, 197), size: 65, angle: 304, color: (229, 31, 98)"

# ---------- まずはベースモデル自体の結果を生成して保存する ----------
base_image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=8.0,
    cross_attention_kwargs={"scale": lora_scale},
    generator=torch.manual_seed(0)  # 再現性確保したい場合
).images[0]

base_save_filename = "output_checkpoint-base.png"
base_save_path = os.path.join(output_dir, base_save_filename)
base_image.save(base_save_path)
print(f"ベースモデルで生成した画像を保存しました: {base_save_path}")

# ---------- checkpoint-1000 から 15000 まで1000刻みでループ ----------
for i in range(1000, 15001, 1000):
    checkpoint_path = os.path.join(model_dir, f"checkpoint-{i}")

    # 各 checkpoint にある UNet ウェイトを読み込み
    pipe.unet = UNet2DConditionModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        torch_dtype=torch.float16
    ).to("cuda")

    # 画像生成
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=8.0,
        cross_attention_kwargs={"scale": lora_scale},
        generator=torch.manual_seed(0)
    ).images[0]

    # ファイル名に checkpoint 番号を付けて保存
    save_filename = f"output_checkpoint-{i}.png"
    save_path = os.path.join(output_dir, save_filename)
    image.save(save_path)

    print(f"checkpoint-{i} で生成した画像を保存しました: {save_path}")

print("すべて完了しました。")
