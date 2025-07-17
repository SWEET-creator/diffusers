import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model
import torchvision.transforms as transforms
from alleta_data import MyDataset

# ベースモデルの読み込み
model_id = "aipicasso/emi-3"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# UNet のパラメータを固定
for param in pipe.unet.parameters():
    param.requires_grad = False

# LoRA のハイパーパラメータ設定（例として、対象モジュールを attention の一部に設定）
lora_config = LoraConfig(
    r=4,               # 低ランク行列のランク
    lora_alpha=16,     # スケーリング係数
    lora_dropout=0.05, # ドロップアウト率
    target_modules=["to_q", "to_v"]  # 適用対象のモジュール（モデルにより異なる）
)

# UNet に LoRA モジュールを注入
pipe.unet = get_peft_model(pipe.unet, lora_config)

import torch
from torch.utils.data import DataLoader


# 前処理（例: リサイズ、テンソル変換、正規化）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# JSONファイルのパスを指定してデータセットを作成
json_file = "dataset/alleta.json"
dataset = MyDataset(json_file, transform=transform)

# ここでは仮に dataset として用意されたデータセットを使用
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    for batch in dataloader:
        images, prompts = batch  # 画像とテキストプロンプトのペア
        
        # 画像とテキストの前処理（トークナイズ、正規化など）を実施
        # 以下は概念的な処理。実際は diffusers のトークナイザや前処理関数を使用してください。
        
        loss = pipe.training_step(images, prompts)  # あくまで疑似コードです
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(pipe.unet.state_dict(), "lora_weights.pt")