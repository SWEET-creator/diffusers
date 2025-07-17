import os
from PIL import Image, ImageOps

def is_gray(color, threshold=100):
    """
    ピクセルがグレーかどうかを判定
    :param color: RGBタプル
    :param threshold: グレー判定の閾値
    :return: グレーなら True, それ以外は False
    """
    return color == (130,130,130)

def apply_gray_to_white_filter(image, threshold=10):
    """
    グレーの部分を白に変更するフィルターを適用
    :param image: PIL.Image オブジェクト
    :param threshold: グレー判定の閾値
    :return: フィルター適用後の画像
    """
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            if is_gray(pixels[x, y], threshold):
                pixels[x, y] = (255, 255, 255)
    return image

def resize_and_pad_image(input_path, output_path, gray_threshold=10):
    # 元の画像を開く
    image = Image.open(input_path).convert("RGB")
    
    # グレー部分を白に変換
    image = apply_gray_to_white_filter(image, gray_threshold)
    
    # 縦を1024ピクセルにリサイズし、アスペクト比を保持する
    aspect_ratio = image.width / image.height
    new_height = 1024
    new_width = int(aspect_ratio * new_height)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 背景を白で埋めた1024x1024の正方形画像を作成
    output_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    
    # 中央に配置する
    offset = ((1024 - new_width) // 2, 0)  # 横の中心に配置
    output_image.paste(resized_image, offset)
    
    # 結果を保存
    output_image.save(output_path)

def process_images_in_directory(input_dir, output_dir, gray_threshold=10):
    """
    指定されたディレクトリ内のPNG画像に対して処理を行う
    :param input_dir: 入力ディレクトリ
    :param output_dir: 出力ディレクトリ
    :param gray_threshold: グレー判定の閾値
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".png"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            print(f"Processing: {input_path}")
            resize_and_pad_image(input_path, output_path, gray_threshold)
            print(f"Saved: {output_path}")

# 使用例
input_dir = "/root/code/diffusers/dataset/lora_alleta_3"  # 入力ディレクトリのパス
output_dir = "/root/code/diffusers/dataset/lora_alleta_3_pad"  # 出力ディレクトリのパス

process_images_in_directory(input_dir, output_dir, gray_threshold=10)
