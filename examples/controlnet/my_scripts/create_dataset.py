import csv
import json
import os

# 入力パスと出力パスの指定
csv_path = "/mnt/data/train_all/metadata.csv"
output_path = "/mnt/data/diffusion-datasets/controlnet-exp1/train.jsonl"
image_base_path = "images/"
conditioning_image_base_path = "conditioning_images/"

# JSONLファイルに書き込む
with open(csv_path, mode='r') as csv_file, open(output_path, mode='w') as jsonl_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        file_name = row['file_name']
        text = row['text']
        
        # JSON形式に必要な情報を追加
        entry = {
            "text": text,
            "image": os.path.join(image_base_path, file_name),
            "conditioning_image": os.path.join(conditioning_image_base_path, file_name)
        }
        
        # JSONL形式で1行ずつ書き込む
        jsonl_file.write(json.dumps(entry) + '\n')

print("metadata.jsonl が作成されました:", output_path)
