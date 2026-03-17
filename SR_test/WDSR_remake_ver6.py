#最终选择进行2x套4x的方式

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torch.nn.functional as F
# 路径设置
high_res_folder = 'E:/USP/test/test_data'  # 高分辨率图像
bicubic_output_folder = 'E:/USP/biCubicConv'  # Bicubic插值输出路径
bicubic_output_folder_2x_se = 'E:/USP/biCubicInter2x'
bicubic_output_folder_8x = 'E:/USP/biCubicConv_8x'
wdsr_output_folder = 'E:/USP/WDSR'  # WDSR超分结果输出路径
wdsr_output_folder_2x_se = 'E:/USP/WDSR_2x_se'
wdsr_output_folder_8x = 'E:/USP/WDSR_8x'
wdsr_output_folder_2x4x = 'E:/USP/WDSR_2x4x'
save_csv_path = 'super_res_results_less_2x2x.csv'  # 保存PSNR和SSIM结果
model_save_path = 'wdsr_trained_model_2x2x.pth'  # 保存训练模型的路径
new_test_folder = 'E:/USP/test/test_data'  # 新的测试数据集
# 确保输出路径存在
os.makedirs(bicubic_output_folder, exist_ok=True)
os.makedirs(wdsr_output_folder, exist_ok=True)

# 上采样倍率
scale = 2

#=============== 双三次插值阶段 ====================
for filename in tqdm(os.listdir(high_res_folder)):  # 使用高分辨率图像作为原图
    img_path = os.path.join(high_res_folder, filename)
    img = cv2.imread(img_path)

    # 下采样再插值回原始大小
    h, w = img.shape[:2]
    low_res = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)  # 从高分辨率图像下采样
    upscaled_img = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)  # 使用双三次插值恢复为原尺寸

    # 保存插值结果
    save_path = os.path.join(bicubic_output_folder_2x_se, filename)
    cv2.imwrite(save_path, upscaled_img)


# =============== WDSR 模型定义 ====================
class WideActivationResidualBlock(nn.Module):
    def __init__(self, in_channels, expansion=6, scaling_factor=0.1):
        super(WideActivationResidualBlock, self).__init__()
        self.scaling_factor = scaling_factor
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm(out)
        return residual + self.scaling_factor * out


class CustomWDSR(nn.Module):
    def __init__(self, num_blocks=8, in_channels=3, scale=scale):
        super(CustomWDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        kernel_size = 3 if scale == 2 else 5
        self.res_blocks = nn.Sequential(
            *[WideActivationResidualBlock(64) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
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


transform = transforms.Compose([
    transforms.CenterCrop((444, 580)),  # 裁剪到 scale=4 的整数倍
    transforms.ToTensor()
])

# =============== 数据集加载 ====================
class BicubicDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_list[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.image_list[idx]


# =============== 动态填充函数 ====================
def pad_to_max(tensor_list):
    max_height = max([t.shape[1] for t in tensor_list])
    max_width = max([t.shape[2] for t in tensor_list])

    padded_tensors = []
    for t in tensor_list:
        pad_height = max_height - t.shape[1]
        pad_width = max_width - t.shape[2]
        padded_tensor = F.pad(t, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=0)


# =============== 自定义collate函数 ====================
def custom_collate_fn(batch):
    imgs, filenames = zip(*batch)
    imgs = pad_to_max(imgs)
    return imgs, filenames


# =============== 训练WDSR模型 ====================
def train_wdsr_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    train_loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for img, _ in tqdm(train_loader):
            img = img.cuda()
            hr_img = img
            lr_img = nn.functional.interpolate(hr_img, scale_factor=1 / scale, mode='bilinear',
                                               align_corners=False)

            output = model(lr_img)
            output = pad_to_max([output, hr_img])[0]  # 确保尺寸一致
            loss = criterion(output, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')

    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('train_loss_curve.png')
    plt.show()


# 初始化模型、损失函数和优化器
model = CustomWDSR().cuda()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 数据加载器
train_dataset = BicubicDataset(bicubic_output_folder_2x_se, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# =============== 主体流程 ====================
TRAIN = False

if TRAIN:
    # 训练模型
    num_epochs = 10  # 设置你需要的训练epoch
    train_wdsr_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)

else:
    # 加载训练好的模型并推理
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # 加载新的测试数据集并裁剪后再插值
    
    def preprocess_test_image(new_test_folder, scale=8):
        img = Image.open(new_test_folder)
        img = transforms.CenterCrop((444, 580))(img)  # 裁剪到 scale 的整数倍
        img = transforms.ToTensor()(img)

        # 下采样和双三次插值恢复
        low_res = nn.functional.interpolate(img.unsqueeze(0), scale_factor=1 / 2, mode='bicubic',
                                            align_corners=False)
        upscaled_img = nn.functional.interpolate(low_res, scale_factor=2, mode='bicubic', align_corners=False)
        return upscaled_img.squeeze(0)  # 恢复成3维图像


    test_images = os.listdir(new_test_folder)
    results = []

    with torch.no_grad():
        for filename in tqdm(test_images):
            img_path = os.path.join(new_test_folder, filename)

            # 图像预处理
            bicubic_img = preprocess_test_image(img_path).cuda()
            lr_img = nn.functional.interpolate(bicubic_img.unsqueeze(0), scale_factor=1 / scale, mode='bilinear',align_corners=False)
            # 通过WDSR模型进行超分辨率重建
            output = model(lr_img)
            output = output.cpu().squeeze(0).permute(1, 2, 0).numpy()

            # 保存WDSR输出结果
            save_path = os.path.join(wdsr_output_folder_2x_se, filename)
            cv2.imwrite(save_path, (output * 255).astype('uint8'))

    # =============== PSNR 和 SSIM 评估 ====================
    for filename in os.listdir(wdsr_output_folder_8x):
        sr_img = cv2.imread(os.path.join(wdsr_output_folder_2x4x, filename))
        hr_img = cv2.imread(os.path.join(new_test_folder, filename))
        hr_img = hr_img[:444, :580]
        print(f"SR Image Shape: {sr_img.shape}, HR Image Shape: {hr_img.shape}")


        #hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]))  # 保证对比时尺寸一致
        psnr_value = psnr(hr_img, sr_img, data_range=255)
        ssim_value = ssim(hr_img, sr_img, data_range=255, channel_axis=2)
        results.append({'Filename': filename, 'PSNR': psnr_value, 'SSIM': ssim_value})

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f'Results saved to {save_csv_path}')
