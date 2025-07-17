import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像の読み込み（カラー：BGR→RGB変換）
img = cv2.imread('/root/code/diffusers/alleta/front.png', cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("入力画像 'front.png' が見つかりません。")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, channels = img_rgb.shape
crow, ccol = rows // 2, cols // 2

# 低周波と高周波をそれぞれ抽出するためのカットオフ周波数
low_radius = 20   # 低周波成分として保持する円の半径
high_radius = 90  # 高周波成分として保持する範囲は外側、内側(high_radius未満)は捨てる

# 2. マスクの作成
# 低周波マスク：中心から low_radius 内の成分を保持
mask_low = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_low, (ccol, crow), low_radius, 1, thickness=-1)

# 高周波マスク：中心から high_radius 未満の部分を捨て、外側を保持
mask_high = np.ones((rows, cols), np.uint8)
cv2.circle(mask_high, (ccol, crow), high_radius, 0, thickness=-1)

# ※ このとき、low_radius < high_radius とすることで
#  中間の周波数（low_radius 以上 high_radius 未満）が捨てられます

# 3. 出力用配列の初期化
img_low = np.zeros((rows, cols, channels), dtype=np.float32)
img_high = np.zeros((rows, cols, channels), dtype=np.float32)
img_recombined = np.zeros((rows, cols, channels), dtype=np.float32)

# 4. 各カラーチャンネルごとにフーリエ変換とフィルタ処理
for ch in range(channels):
    # フーリエ変換と中心化
    f = np.fft.fft2(img_rgb[:, :, ch])
    fshift = np.fft.fftshift(f)
    
    # 低周波成分のみ抽出
    fshift_low = fshift * mask_low
    # 高周波成分のみ抽出（中間領域は含まない）
    fshift_high = fshift * mask_high
    
    # 低周波成分の逆変換
    f_ishift_low = np.fft.ifftshift(fshift_low)
    low_channel = np.fft.ifft2(f_ishift_low)
    low_channel = np.abs(low_channel)
    
    # 高周波成分の逆変換
    f_ishift_high = np.fft.ifftshift(fshift_high)
    high_channel = np.fft.ifft2(f_ishift_high)
    high_channel = np.abs(high_channel)
    
    # 再合成（低周波＋高周波のみ足し合わせる＝中間周波は捨てる）
    fshift_recombined = fshift_low + fshift_high
    f_ishift_recombined = np.fft.ifftshift(fshift_recombined)
    recombined_channel = np.fft.ifft2(f_ishift_recombined)
    recombined_channel = np.abs(recombined_channel)
    
    img_low[:, :, ch] = low_channel
    img_high[:, :, ch] = high_channel
    img_recombined[:, :, ch] = recombined_channel

# 5. 値を0〜255にクリップしてuint8に変換
img_low = np.uint8(np.clip(img_low, 0, 255))
img_high = np.uint8(np.clip(img_high, 0, 255))
img_recombined = np.uint8(np.clip(img_recombined, 0, 255))

# 6. 結果の表示と保存
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
plt.title('Recombined Image (Mid discarded)')
plt.axis('off')

plt.tight_layout()
plt.savefig('output.jpg')
plt.show()
