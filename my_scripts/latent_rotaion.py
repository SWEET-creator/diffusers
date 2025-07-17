import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

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

# ---- ノイズを n 度回転させる処理 ----
n_degrees = 90  # 回転角度（例: 45度）
theta = math.radians(n_degrees)

# 回転行列の作成（アフィン変換用）
rotation_matrix = torch.tensor([
    [math.cos(theta), -math.sin(theta), 0],
    [math.sin(theta),  math.cos(theta), 0]
], device=original_noise.device, dtype=original_noise.dtype).unsqueeze(0)  # shape: (1, 2, 3)

# アフィングリッドの生成
grid = F.affine_grid(rotation_matrix, original_noise.size(), align_corners=False)

# ノイズの回転
rotated_noise = F.grid_sample(original_noise, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
# ------------------------------------

# テキストプロンプトの定義
prompt = "1boy"

# 元のノイズから画像生成
result_original = pipe(prompt, latents=original_noise, num_inference_steps=50, guidance_scale=7.5)
image_original = result_original.images[0]

# 回転後のノイズから画像生成
result_rotated = pipe(prompt, latents=rotated_noise, num_inference_steps=50, guidance_scale=7.5)
image_rotated = result_rotated.images[0]

# 結果の比較表示
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image_original)
axs[0].set_title("Original Noise")
axs[0].axis("off")
axs[1].imshow(image_rotated)
axs[1].set_title(f"Noise Rotated by {n_degrees}°")
axs[1].axis("off")
plt.tight_layout()
plt.savefig("result_comparison.png")
