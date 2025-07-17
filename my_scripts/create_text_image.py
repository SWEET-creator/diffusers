from PIL import Image, ImageDraw, ImageFont

# 画像を読み込む
image = Image.open("/root/code/diffusers/output/lain.png")

# 描画用のオブジェクトを作成
draw = ImageDraw.Draw(image)

# 挿入するテキスト
text = "This image is protected by copyright and editing is prohibited."

# フォントを指定（使用するフォントファイルのパスとサイズを指定）
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=30)

# テキストの描画位置と色（RGB形式）を設定
position = (50, 50)
color = (255, 255, 255)  # 白色

# テキストを画像に描画
draw.text(position, text, fill=color, font=font)

# 文字が挿入された画像を保存
image.save("example_with_text_2_eng.jpg")
