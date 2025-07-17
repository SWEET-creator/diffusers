import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像の読み込み（カラー：BGRとして読み込まれるためRGBに変換）
img = cv2.imread('/root/code/diffusers/alleta/front.png', cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("入力画像 'front.png' が見つかりません。")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画像サイズとチャンネル数の取得
rows, cols, channels = img_rgb.shape

# 2. 低周波・高周波フィルタ用のマスク作成（全チャンネル共通）
crow, ccol = rows // 2, cols // 2
radius = 30  # カットオフ周波数（調整可能）
mask_low = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_low, (ccol, crow), radius, 1, thickness=-1)
mask_high = 1 - mask_low  # 低周波マスクの補集合

# 各チャンネルごとに処理するための出力配列（floatで作業）
img_low = np.zeros((rows, cols, channels), dtype=np.float32)
img_high = np.zeros((rows, cols, channels), dtype=np.float32)
img_recombined = np.zeros((rows, cols, channels), dtype=np.float32)

# 3. 各カラーチャンネルごとにフーリエ変換とフィルタ処理
for ch in range(channels):
    # フーリエ変換と中心化
    f = np.fft.fft2(img_rgb[:, :, ch])
    fshift = np.fft.fftshift(f)
    
    # マスク適用
    fshift_low = fshift * mask_low
    fshift_high = fshift * mask_high
    
    # 低周波画像の逆変換
    f_ishift_low = np.fft.ifftshift(fshift_low)
    low_channel = np.fft.ifft2(f_ishift_low)
    low_channel = np.abs(low_channel)
    
    # 高周波画像の逆変換
    f_ishift_high = np.fft.ifftshift(fshift_high)
    high_channel = np.fft.ifft2(f_ishift_high)
    high_channel = np.abs(high_channel)
    
    # 再合成（周波数領域での足し合わせ→逆変換）
    fshift_recombined = fshift_low + fshift_high
    f_ishift_recombined = np.fft.ifftshift(fshift_recombined)
    recombined_channel = np.fft.ifft2(f_ishift_recombined)
    recombined_channel = np.abs(recombined_channel)
    
    # 結果を保存
    img_low[:, :, ch] = low_channel
    img_high[:, :, ch] = high_channel
    img_recombined[:, :, ch] = recombined_channel

# 4. 値を0〜255の範囲にクリップし、uint8に変換
img_low = np.uint8(np.clip(img_low, 0, 255))
img_high = np.uint8(np.clip(img_high, 0, 255))
img_recombined = np.uint8(np.clip(img_recombined, 0, 255))

# 5. 結果の表示（MatplotlibはRGB形式）
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_low)
plt.title('Low-Pass Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_high)
plt.title('High-Pass Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_recombined)
plt.title('Recombined Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('output.jpg')  # plt.show()の前に保存
plt.show()
