import os
import random
import shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np

# 路径设置
high_res_folder = 'E:/USPPP/'
bicubic_output_folder_final = 'E:/USP/biCubicInter_final'
wdsr_output_folder_final = 'E:/USP/WDSR_final'
save_csv_path = 'super_res_results_final.csv'
pretrained_weight_path = 'wdsr-b-32-x4.pth'

os.makedirs(bicubic_output_folder_final, exist_ok=True)
os.makedirs(wdsr_output_folder_final, exist_ok=True)

bicubic_scale = 2
wdsr_scale = 4


# 获取图像路径
def get_image_paths(folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# 双立方插值
def perform_bicubic_interpolation():
    if os.path.exists(bicubic_output_folder_final) and len(get_image_paths(bicubic_output_folder_final)) > 0:
        print("Bicubic interpolation already performed. Skipping...")
        return
    print("Performing 2x Bicubic Interpolation...")
    image_paths = get_image_paths(high_res_folder)
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        low_res = cv2.resize(img, (w // bicubic_scale, h // bicubic_scale), interpolation=cv2.INTER_AREA)
        upscaled_img = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        relative_path = os.path.relpath(img_path, high_res_folder)
        save_path = os.path.join(bicubic_output_folder_final, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, upscaled_img)


# 数据集划分
def split_dataset(image_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "比例总和必须为1"
    train_folder = os.path.join(image_folder, 'train')
    val_folder = os.path.join(image_folder, 'val')
    test_folder = os.path.join(image_folder, 'test')
    if (os.path.exists(train_folder) and os.path.exists(val_folder) and os.path.exists(test_folder) and
            len(get_image_paths(train_folder)) > 0 and len(get_image_paths(val_folder)) > 0 and len(
                get_image_paths(test_folder)) > 0):
        print("Dataset already split. Skipping...")
        return train_folder, val_folder, test_folder
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    image_paths = get_image_paths(image_folder)
    random.shuffle(image_paths)
    num_images = len(image_paths)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)
    for i, img_path in enumerate(image_paths):
        if i < train_end:
            dest_folder = train_folder
        elif i < val_end:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        relative_path = os.path.relpath(img_path, image_folder)
        dest_path = os.path.join(dest_folder, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(img_path, dest_path)
    print(f"数据集划分完成：训练集 ({train_ratio * 100}%), 验证集 ({val_ratio * 100}%), 测试集 ({test_ratio * 100}%)")
    return train_folder, val_folder, test_folder


# WDSR-B 模型定义
class WDSRResidualBlock(nn.Module):
    def __init__(self, num_filters=32, expansion=6, linear=0.8):
        super(WDSRResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters * expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters * expansion, int(num_filters * linear), kernel_size=1)
        self.conv3 = nn.Conv2d(int(num_filters * linear), num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return residual + out


class WDSR_B(nn.Module):
    def __init__(self, scale=4, num_filters=32, num_res_blocks=32):
        super(WDSR_B, self).__init__()
        self.scale = scale
        self.num_filters = num_filters
        self.main_conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[WDSRResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
        self.main_conv2 = nn.Conv2d(num_filters, 3 * scale ** 2, kernel_size=3, padding=1)
        self.main_upsample = nn.PixelShuffle(scale)
        self.skip_conv = nn.Conv2d(3, 3 * scale ** 2, kernel_size=5, padding=2)
        self.skip_upsample = nn.PixelShuffle(scale)

    def forward(self, x):
        main = self.main_conv1(x)
        main = self.res_blocks(main)
        main = self.main_conv2(main)
        main = self.main_upsample(main)
        skip = self.skip_conv(x)
        skip = self.skip_upsample(skip)
        return main + skip


# 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop((444, 580)),
    transforms.ToTensor()
])


# 数据集加载
class BicubicDataset(Dataset):
    def __init__(self, image_folder, transform=None, is_test=False):
        self.image_paths = get_image_paths(image_folder)
        self.transform = transform
        self.is_test = is_test  # 用于区分测试集和训练/验证集

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.is_test:
            # 测试集：加载灰度图像并转换为伪 RGB
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            pseudo_rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(pseudo_rgb_img)
        else:
            # 训练/验证集：直接加载 RGB 图像
            img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, img_path


# 主体流程
perform_bicubic_interpolation()
train_folder, val_folder, test_folder = split_dataset(bicubic_output_folder_final, train_ratio=0.7, val_ratio=0.2,
                                                      test_ratio=0.1)

model = WDSR_B(scale=wdsr_scale, num_filters=32, num_res_blocks=32).cuda()
state_dict = torch.load(pretrained_weight_path)
print("Validating weights...")
for key, value in state_dict.items():
    if torch.isnan(value).any() or torch.isinf(value).any():
        print(f"Weight {key} contains NaN or Inf!")
    else:
        print(f"Weight {key} is valid, min: {value.min().item()}, max: {value.max().item()}")
model.load_state_dict(state_dict)
model.eval()

print("Testing model with sample input...")
test_input = torch.ones(1, 3, 111, 145).cuda() * 0.5
with torch.no_grad():
    test_output = model(test_input)
    if torch.isnan(test_output).any() or torch.isinf(test_output).any():
        print("Test output contains NaN or Inf!")
    else:
        print(
            f"Test output shape: {test_output.shape}, min: {test_output.min().item()}, max: {test_output.max().item()}")

train_dataset = BicubicDataset(train_folder, transform=transform, is_test=False)
val_dataset = BicubicDataset(val_folder, transform=transform, is_test=False)
test_dataset = BicubicDataset(test_folder, transform=transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Starting WDSR-B inference...")
results = []
with torch.no_grad():
    for batch_img, batch_img_paths in tqdm(test_loader, desc="Testing"):
        batch_img = batch_img.cuda()
        print(f"Batch image min: {batch_img.min().item()}, max: {batch_img.max().item()}")

        lr_img = F.interpolate(batch_img, scale_factor=1 / wdsr_scale, mode='bicubic', align_corners=False)
        print(f"Low-res image min: {lr_img.min().item()}, max: {lr_img.max().item()}")

        output = model(lr_img)
        if torch.isnan(output).any():
            print("Model output contains NaN! Skipping this batch.")
            continue
        output = output.clamp(0, 1)
        print(f"Output min: {output.min().item()}, max: {output.max().item()}")

        for i in range(output.size(0)):
            single_output = output[i].permute(1, 2, 0).cpu().numpy()
            single_output = (single_output * 255).astype('uint8')
            print(f"Single output shape: {single_output.shape}, min: {single_output.min()}, max: {single_output.max()}")

            # 转换为灰度图像并保存
            gray_output = cv2.cvtColor(single_output, cv2.COLOR_RGB2GRAY)
            relative_path = os.path.relpath(batch_img_paths[i], test_folder)
            save_path = os.path.join(wdsr_output_folder_final, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, gray_output)

            # 加载保存的灰度图像并转换为三通道以计算指标
            wdsr_gray = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
            wdsr_img = cv2.cvtColor(wdsr_gray, cv2.COLOR_GRAY2BGR)  # 转换为 BGR 以匹配 hr_img 格式

            # 加载原始高分辨率图像
            original_img_path = os.path.join(high_res_folder, relative_path)
            hr_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
            if hr_img is None:
                print(f"Failed to load original image: {original_img_path}")
                continue

            # 调整原始图像大小并转换为灰度后转为 BGR
            hr_img = cv2.resize(hr_img, (wdsr_img.shape[1], wdsr_img.shape[0]))
            hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
            hr_img = cv2.cvtColor(hr_gray, cv2.COLOR_GRAY2BGR)  # 保持与 wdsr_img 一致

            # 计算 PSNR 和 SSIM
            psnr_value = psnr(hr_img, wdsr_img, data_range=255)
            ssim_value = ssim(hr_img, wdsr_img, data_range=255, channel_axis=2)
            results.append({'Filename': relative_path, 'PSNR': psnr_value, 'SSIM': ssim_value})

df = pd.DataFrame(results)
df.to_csv(save_csv_path, index=False)
print(f"Results saved to {save_csv_path}")