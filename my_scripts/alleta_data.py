import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, json_file, transform=None):
        # JSONファイルを読み込む
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]  # 画像ファイルのパス
        prompt = sample["prompt"]          # テキストプロンプト

        # 画像を読み込み、RGBに変換
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, prompt
