import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

import  pandas as pd
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




class PreprocessDataset(Dataset):
    def __init__(self, img_dir, split='train', transform=transform, val_ratio=0.2):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]

        # 数据划分
        self.train_imgs, self.val_imgs = train_test_split(self.imgs, test_size=val_ratio, random_state=42)

        if split == 'train':
            self.imgs = self.train_imgs
        elif split == 'val':
            self.imgs = self.val_imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        low_res = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.25, mode='bicubic').squeeze(0)
        return low_res, img

# 自定义数据集类
#class PreprocessDataset(Dataset):
#    def __init__(self, img_dir, transform=transform):
#        self.img_dir = img_dir
#        self.transform = transform
#        self.imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]

#    def __len__(self):
#        return len(self.imgs)

#    def __getitem__(self, idx):
#        img_path = self.imgs[idx]
#        img = Image.open(img_path)
#        if self.transform:
#            img = self.transform(img)
#        low_res = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.25, mode='bicubic').squeeze(0)
#        return low_res, img


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = WDSR_SRResNet().to(device)

# 判断是否进行训练或者评估
TRAIN = False  # 如果为 True，则进行训练；如果为 False，则跳过训练并加载已保存的模型

if TRAIN:
    # 训练模型
    if TRAIN:
        train_dataset = PreprocessDataset('E:/USP/All', split='train', transform=transform)
        val_dataset = PreprocessDataset('E:/USP/All', split='val', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        EPOCHS = 20

        # 记录损失的列表
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch, (low_res, high_res) in enumerate(train_loader):
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                optimizer.zero_grad()
                outputs = model(low_res)
                loss = criterion(outputs, high_res)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)  # 记录训练损失
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_loss:.4f}')

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for low_res, high_res in val_loader:
                    low_res = low_res.to(device)
                    high_res = high_res.to(device)
                    outputs = model(low_res)
                    loss = criterion(outputs, high_res)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)  # 记录验证损失
            print(f'Validation Loss: {avg_val_loss:.4f}')

            # 绘制训练损失曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_loss_curve.png')  # 保存训练损失图像
        plt.show()

        # 绘制验证损失曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('val_loss_curve.png')  # 保存验证损失图像
        plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'wdsr_srresnet_model20.pth')
    print("Model saved successfully!")
else:
    # 加载已保存的模型
    model.load_state_dict(torch.load('wdsr_srresnet_model20.pth'))
    model.eval()  # 设置为评估模式
    print("Model loaded successfully for evaluation!")

    # 加载新的数据集并进行 PSNR 和 SSIM 评估
    new_data_dir = 'E:/USP/test/frame_crop/'
    new_dataset = PreprocessDataset(new_data_dir, transform=transform)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False)

    psnr_ssim_results = []
    with torch.no_grad():
        for idx, (low_res, high_res) in enumerate(new_loader):
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            outputs = model(low_res)
            outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
            high_res = high_res.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(outputs.shape[0]):
                psnr_value = psnr(outputs[i], high_res[i], data_range=1.0)
                ssim_value = ssim(outputs[i], high_res[i], data_range=1.0, channel_axis=2)
                psnr_ssim_results.append({
                    "Image_Index": idx * new_loader.batch_size + i,
                    "PSNR": psnr_value,
                    "SSIM": ssim_value
                })
    df = pd.DataFrame(psnr_ssim_results)
    df.to_csv("psnr_ssim_results_new_data.csv", index=False)
    print("PSNR and SSIM for each image have been saved to 'psnr_ssim_results_new_data.csv'")