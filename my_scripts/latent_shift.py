import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# モデルIDの指定
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# モデルの読み込み（半精度 fp16 を使用）
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")  # GPU を使用

# 生成画像サイズと潜在空間の設定
height, width = 512, 512
scale_factor = 8  # VAE によるダウンサンプリングスケール（通常は8）
latent_shape = (1, 4, height // scale_factor, width // scale_factor)

# シード固定のためのジェネレーター（例: 42）
generator = torch.Generator("cuda").manual_seed(42)

# 初期ノイズの生成（torch.float16 を指定）
original_noise = torch.randn(latent_shape, generator=generator, device="cuda", dtype=torch.float16)

# 横方向にシフト（例: 10ピクセル）させたノイズを作成
shift_pixels = 10
shifted_noise = torch.roll(original_noise, shifts=shift_pixels, dims=-1)  # 横方向（最後の次元）をシフト

# テキストプロンプトの定義
prompt = "1girl"

# 元のノイズから画像生成
result_original = pipe(prompt, latents=original_noise, num_inference_steps=50, guidance_scale=7.5)
image_original = result_original.images[0]

# シフト後のノイズから画像生成
result_shifted = pipe(prompt, latents=shifted_noise, num_inference_steps=50, guidance_scale=7.5)
image_shifted = result_shifted.images[0]

# 結果の比較表示
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image_original)
axs[0].set_title("Original Noise")
axs[0].axis("off")
axs[1].imshow(image_shifted)
axs[1].set_title(f"Noise Shifted by {shift_pixels} px")
axs[1].axis("off")
plt.tight_layout()
plt.savefig("result_comparison.png")