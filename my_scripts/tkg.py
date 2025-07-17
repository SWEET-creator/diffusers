import torch
import torch.nn.functional as F
import math
import numpy as np
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# --- 前準備（既存コード） ---
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

# --- ここからが実装の続き ---

# 1. チャネル平均シフト（Channel Mean Shift）の実装
def channel_mean_shift(noise, target_shifts, step=0.001, max_iter=1000):
    """
    noise: tensor of shape (batch, channels, H, W)
    target_shifts: 各チャネルで増加させたい正の比率（例: [0, 0.07, 0.07, 0]）
    noise内の各チャネルについて、正の値の割合が target_shift 分だけ増えるように定数シフトを適用する
    """
    # ※演算精度のため、一旦float32にキャスト（最終的にfloat16に戻します）
    noise = noise.to(torch.float32)
    noise_shifted = noise.clone()
    batch, channels, H, W = noise.shape

    for c in range(channels):
        target = target_shifts[c]
        if target != 0:
            # 現在のチャネルの正の割合
            channel_data = noise_shifted[0, c, :, :]
            initial_ratio = (channel_data > 0).float().mean().item()
            target_ratio = initial_ratio + target

            shift_val = 0.0
            iter_count = 0
            # 目標の正の割合に到達するまで、シフト量を徐々に増加
            while iter_count < max_iter:
                shifted_channel = channel_data + shift_val
                positive_ratio = (shifted_channel > 0).float().mean().item()
                if positive_ratio >= target_ratio:
                    noise_shifted[0, c, :, :] = shifted_channel
                    break
                shift_val += step
                iter_count += 1
            if iter_count == max_iter:
                print(f"Warning: max iterations reached for channel {c}")
    
    return noise_shifted.to(torch.float16)

# 2. ガウスマスクの生成関数
def create_gaussian_mask(height, width, mu_i, mu_j, sigma):
    """
    height, width: マスクのサイズ（潜在空間サイズ）
    mu_i, mu_j: マスクの中心位置（例: height/2, width/2）
    sigma: ガウス分布の広がり（小さいほど中央部のみ1に近く、外側は0に近い）
    """
    # 各座標のグリッドを作成
    y = torch.arange(height, dtype=torch.float32, device="cuda").unsqueeze(1)  # (height, 1)
    x = torch.arange(width, dtype=torch.float32, device="cuda").unsqueeze(0)   # (1, width)
    mask = torch.exp(-(((x - mu_j)**2 + (y - mu_i)**2) / (2 * sigma**2)))
    # マスクを (1,1,H,W) に整形（チャネル方向にブロードキャスト可能に）
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

# 3. ターゲットシフトの設定
# ※本実験では、緑背景（クロマキー）を得るため、チャネル1（2番目）とチャネル2（3番目）に +7% のシフトを適用
target_shifts = [0.0, 0.07, 0.07, 0.0]

# 4. チャネル平均シフトを適用して init color noise を生成
shifted_noise = channel_mean_shift(original_noise, target_shifts)

# 5. ガウスマスクの生成（潜在空間サイズに合わせる）
latent_H = latent_shape[2]
latent_W = latent_shape[3]
mu_i = latent_H / 2
mu_j = latent_W / 2
sigma = 10  # foreground領域のサイズを調整（値を大きくするとforegroundが広がります）
gaussian_mask = create_gaussian_mask(latent_H, latent_W, mu_i, mu_j, sigma)

# 6. 元のノイズとカラーシフト済みノイズをガウスマスクでブレンド
#    foreground領域は元のノイズ、background領域はカラーシフト済みノイズを使用
z_T_key = gaussian_mask * original_noise + (1 - gaussian_mask) * shifted_noise

# 7. テキストプロンプトを指定
prompt = "A cat running in a park"  # 例として「公園で走る猫」

# 8. それぞれのlatentから生成結果を取得
with torch.autocast("cuda"):
    # オリジナルノイズからの生成
    result_original = pipe(prompt=prompt, latents=original_noise, guidance_scale=7.5, num_inference_steps=50)
    # チャネルシフトのみ適用したlatentからの生成
    result_shift = pipe(prompt=prompt, latents=shifted_noise, guidance_scale=7.5, num_inference_steps=50)
    # チャネルシフト＋ガウスマスク（TKG-DM）のlatentからの生成
    result_z_key = pipe(prompt=prompt, latents=z_T_key, guidance_scale=7.5, num_inference_steps=50)

img_original = result_original.images[0]
img_shift = result_shift.images[0]
img_z_key = result_z_key.images[0]

# 9. latentのRGB可視化用関数
def latent_to_rgb(latent):
    """
    latent: tensor of shape (1, 4, H, W)
    先頭の3チャネルをRGBとして取り出し、min-max正規化してnumpy配列に変換
    """
    # ここで float32 に変換
    latent_rgb = latent[0, :3, :, :].to(torch.float32)  # (3, H, W)
    # 全体での最小・最大を取得して正規化
    min_val = latent_rgb.min()
    max_val = latent_rgb.max()
    latent_rgb = (latent_rgb - min_val) / (max_val - min_val + 1e-8)
    latent_rgb = latent_rgb.detach().cpu().numpy()
    # 転置して (H, W, 3) に
    latent_rgb = np.transpose(latent_rgb, (1, v, 0))
    return latent_rgb


rgb_original = latent_to_rgb(original_noise)
rgb_shifted = latent_to_rgb(shifted_noise)
rgb_z_key = latent_to_rgb(z_T_key)

# 10. 6枚の結果を1枚の図にまとめて表示・保存
# 上段：latentのRGB可視化、下段：生成画像
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# 上段：latentのRGB
axs[0, 0].imshow(rgb_original)
axs[0, 0].set_title("Original Latent (RGB)")
axs[0, 0].axis("off")

axs[0, 1].imshow(rgb_shifted)
axs[0, 1].set_title("Channel Shift Latent (RGB)")
axs[0, 1].axis("off")

axs[0, 2].imshow(rgb_z_key)
axs[0, 2].set_title("Shift+Gaussian Mask Latent (RGB)")
axs[0, 2].axis("off")

# 下段：生成結果
axs[1, 0].imshow(img_original)
axs[1, 0].set_title("Generation with Original Noise")
axs[1, 0].axis("off")

axs[1, 1].imshow(img_shift)
axs[1, 1].set_title("Generation with Channel Shift Only")
axs[1, 1].axis("off")

axs[1, 2].imshow(img_z_key)
axs[1, 2].set_title("Generation with Shift+Gaussian Mask (TKG-DM)")
axs[1, 2].axis("off")

plt.tight_layout()
plt.savefig("result_image_tkg.png")
plt.show()
