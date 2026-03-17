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
import pandas as pd

# 数据预处理，随机裁剪
transform = transforms.Compose([transforms.RandomCrop(96), transforms.ToTensor()])


# EDSR模型定义
class EDSR(nn.Module):
    def __init__(self, args, conv=nn.Conv2d):
        super(EDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        # 均值和标准差
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # 头部模块
        m_head = [conv(args.n_colors, n_feats, kernel_size, padding=(kernel_size // 2))]  # 添加padding

        # 主体模块
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))  # 添加padding

        # 尾部模块
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size // 2))  # 添加padding
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x  # 保证res和x的尺寸相同

        x = self.tail(res)
        x = self.add_mean(x)

        return x


# 定义ResBlock
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
            act,
            conv(n_feats, n_feats, kernel_size, padding=(kernel_size // 2))
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# 定义图像均值偏移
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


# 定义Upsampler模块
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, act=False):
        m = []
        if (scale & (scale - 1)) == 0:  # 如果scale是2的倍数
            for _ in range(int(np.log2(scale))):
                m.append(conv(n_feats, 4 * n_feats, 3, padding=1))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


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


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 超参数定义
class Args:
    def __init__(self):
        self.n_resblocks = 16
        self.n_feats = 64
        self.scale = [4]
        self.rgb_range = 255
        self.n_colors = 3
        self.res_scale = 1.0
        self.dilation = False


args = Args()

# 加载模型
model = EDSR(args).to(device)

# 判断是否进行训练或者评估
TRAIN = False # 设置为True进行训练，设置为False加载模型进行评估

if TRAIN:
    # 划分训练集和验证集
    dataset = PreprocessDataset('E:/UltraSPic/jpg_crop/', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 100

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
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {avg_loss:.4f}')

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
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Validation Loss: {avg_val_loss:.4f}')

    # 绘制训练和验证损失曲线
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', color='orange')
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_loss.png')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'edsr_model.pth')
    print("Model saved successfully!")
else:
    # 加载已保存的模型
    model.load_state_dict(torch.load('edsr_model.pth'))
    model.eval()
    print("Model loaded successfully for evaluation!")

    # 新数据集评估
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
                psnr_value = psnr(high_res[i], outputs[i])
                ssim_value = ssim(high_res[i], outputs[i], multichannel=True)
                psnr_ssim_results.append([psnr_value, ssim_value])

    # 保存PSNR和SSIM值
    df = pd.DataFrame(psnr_ssim_results, columns=['PSNR', 'SSIM'])
    df.to_csv('edsr_psnr_ssim_results.csv', index=False)
    print("Evaluation complete. Results saved to edsr_psnr_ssim_results.csv.")
