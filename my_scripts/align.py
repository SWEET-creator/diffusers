import matplotlib.pyplot as plt 
import numpy as np
import os

image_dir = "/root/code/diffusers/output/results/"
image_files = [
    "output_checkpoint-base.png",
    "output_checkpoint-1000.png",
    "output_checkpoint-2000.png",
    "output_checkpoint-3000.png",
    "output_checkpoint-4000.png",
    "output_checkpoint-5000.png",
    "output_checkpoint-6000.png",
    "output_checkpoint-7000.png",
    "output_checkpoint-8000.png",
    "output_checkpoint-9000.png",
    "output_checkpoint-10000.png",
    "output_checkpoint-11000.png",
    "output_checkpoint-12000.png",
    "output_checkpoint-13000.png",
    "output_checkpoint-14000.png",
]
# 画像の読み込み
images = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = plt.imread(image_path)
    images.append(image)
# 画像のサイズを取得
image_shape = images[0].shape
# 画像を表示
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for ax, image_file, image in zip(axes.flatten(), image_files, images):
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(image_file)
plt.tight_layout()
plt.show()
plt.savefig("output.png", dpi=600)