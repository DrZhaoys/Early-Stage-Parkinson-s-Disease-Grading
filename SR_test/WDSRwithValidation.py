import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# 数据预处理，随机裁剪
transform = transforms.Compose([transforms.RandomCrop(96), transforms.ToTensor()])


# 定义带有缩放的残差块（WDSR 风格）
class ScaledResidualBlock(nn.Module):
    def __init__(self, in_channels, expansion=6, scaling_factor=0.1):
        super(ScaledResidualBlock, self).__init__()
        self.scaling_factor = scaling_factor
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return residual + self.scaling_factor * out


# 定义WDSR风格的SRResNet
class WDSR_SRResNet(nn.Module):
    def __init__(self, num_blocks=16, in_channels=3, scale=4):
        super(WDSR_SRResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[ScaledResidualBlock(64) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 上采样模块
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = x + residual
        x = self.upsample(x)
        return x


# 自定义数据集类
class PreprocessDataset(Dataset):
    def __init__(self, img_dir, transform=transform):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        low_res = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.25, mode='bicubic').squeeze(0)
        return low_res, img


# 加载数据集
full_dataset = PreprocessDataset('E:/USP/All/', transform=transform)

# 设置训练集和验证集的划分比例（80%训练，20%验证）
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# 随机划分数据集
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# 加载训练集和验证集
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 设置设备，加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WDSR_SRResNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型，并添加验证损失的计算
EPOCHS = 20
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0

    # 训练阶段
    for batch, (low_res, high_res) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # 验证阶段
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            outputs = model(low_res)
            val_loss = criterion(outputs, high_res)
            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 训练结束后保存模型
torch.save(model.state_dict(), 'wdsr_srresnet_model.pth')
print("Model saved successfully!")

# 绘制训练和验证损失曲线
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
