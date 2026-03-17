import os
import random
import shutil
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
high_res_folder = 'E:/USPPP/'  # 高分辨率图像
bicubic_output_folder_final = 'E:/USP/biCubicInter_final'  # 2倍双立方插值输出路径
wdsr_output_folder_final = 'E:/USP/WDSR_final'  # 4倍WDSR输出路径
save_csv_path = 'super_res_results_final.csv'  # 保存PSNR和SSIM结果
model_save_path = 'wdsr_trained_model_2x4x_final.pth'  # 保存训练模型的路径

# 确保输出路径存在
os.makedirs(bicubic_output_folder_final, exist_ok=True)
os.makedirs(wdsr_output_folder_final, exist_ok=True)

# 上采样倍率
bicubic_scale = 2  # 双立方插值倍率
wdsr_scale = 4     # WDSR超分辨率倍率

# ### 递归获取图像路径
def get_image_paths(folder):
    """递归获取文件夹中所有图像文件的路径"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

# ### 双立方插值阶段 (2倍)
def perform_bicubic_interpolation():
    """执行2倍双立方插值，如果已有结果则跳过"""
    if os.path.exists(bicubic_output_folder_final) and len(get_image_paths(bicubic_output_folder_final)) > 0:
        print("Bicubic interpolation already performed. Skipping...")
        return
    print("Performing 2x Bicubic Interpolation...")
    image_paths = get_image_paths(high_res_folder)
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        low_res = cv2.resize(img, (w // bicubic_scale, h // bicubic_scale), interpolation=cv2.INTER_AREA)
        upscaled_img = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        relative_path = os.path.relpath(img_path, high_res_folder)
        save_path = os.path.join(bicubic_output_folder_final, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, upscaled_img)

# ### 数据集划分
def split_dataset(image_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """将数据集划分为训练集、验证集和测试集，如果已有划分则跳过"""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "比例总和必须为1"
    train_folder = os.path.join(image_folder, 'train')
    val_folder = os.path.join(image_folder, 'val')
    test_folder = os.path.join(image_folder, 'test')
    if (os.path.exists(train_folder) and os.path.exists(val_folder) and os.path.exists(test_folder) and
        len(get_image_paths(train_folder)) > 0 and len(get_image_paths(val_folder)) > 0 and len(get_image_paths(test_folder)) > 0):
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

# ### 加权归一化层
class WeightedNorm(nn.Module):
    def __init__(self, num_features):
        super(WeightedNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)

# ### 宽激活残差块
class WideActivationResidualBlock(nn.Module):
    def __init__(self, in_channels,
                 expansion=6):
        super(WideActivationResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels, kernel_size=3, padding=1)
        self.wn = WeightedNorm(in_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.wn(out)
        return residual + out

# ### Custom-WDSR 模型
class CustomWDSR(nn.Module):
    def __init__(self, num_blocks=8, in_channels=1, scale=4, expansion=6):
        super(CustomWDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.wn1 = WeightedNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[WideActivationResidualBlock(64, expansion) for _ in range(num_blocks)]
        )
        kernel_size = 5 if scale == 4 else 3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.wn2 = WeightedNorm(64)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            WeightedNorm(in_channels)
        )
    def forward(self, x):
        x = self.relu(self.wn1(self.conv1(x)))
        residual = x
        x = self.res_blocks(x)
        x = self.wn2(self.conv2(x))
        x = x + residual
        x = self.upsample(x)
        return x

# ### 自定义对数损失函数（无掩码）
class CustomLogLoss(nn.Module):
    def __init__(self, epsilon=1e-4, k=5):
        super(CustomLogLoss, self).__init__()
        self.epsilon = epsilon  # 避免log(0)
        self.k = k  # 缩放因子
    def forward(self, y, y_hat):
        loss = torch.log((torch.abs(y - y_hat) + self.epsilon) / self.k)
        return loss.mean()  # 全图均值

# ### 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop((444, 576)),  # 调整为能被4整除的尺寸
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])

# ### 数据集加载
class BicubicDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = get_image_paths(image_folder)
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, img_path

# ### 训练WDSR模型
def train_wdsr_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for img, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)"):
            img = img.cuda()
            hr_img = img
            lr_img = nn.functional.interpolate(hr_img, scale_factor=1 / wdsr_scale, mode='bilinear', align_corners=False)
            output = model(lr_img)
            loss = criterion(hr_img, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for img, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)"):
                img = img.cuda()
                hr_img = img
                lr_img = nn.functional.interpolate(hr_img, scale_factor=1 / wdsr_scale, mode='bilinear', align_corners=False)
                output = model(lr_img)
                loss = criterion(hr_img, output)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('train_val_loss_curve.png')
    plt.show()

# ### 主体流程
# 1. 执行双立方插值
perform_bicubic_interpolation()

# 2. 划分数据集
train_folder, val_folder, test_folder = split_dataset(bicubic_output_folder_final, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# 3. 初始化模型、损失函数和优化器
model = CustomWDSR(num_blocks=8, in_channels=1, scale=wdsr_scale).cuda()
criterion = CustomLogLoss().cuda()  # 无掩码对数损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 数据加载器
train_dataset = BicubicDataset(train_folder, transform=transform)
val_dataset = BicubicDataset(val_folder, transform=transform)
test_dataset = BicubicDataset(test_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 5. 设置训练或测试模式
TRAIN = False  # True为训练，False为推理

if TRAIN:
    num_epochs = 300
    train_wdsr_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs)
else:
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    results = []
    with torch.no_grad():

        for batch_img, batch_img_paths in tqdm(test_loader, desc="Testing"):
            batch_img = batch_img.cuda()
            lr_img = nn.functional.interpolate(batch_img, scale_factor=1 / wdsr_scale, mode='bilinear', align_corners=False)
            output = model(lr_img)
            for i in range(output.size(0)):
                single_output = output[i].squeeze(0).cpu().numpy()
                single_output = (single_output * 0.5 + 0.5) * 255  # 反归一化到[0,255]
                relative_path = os.path.relpath(batch_img_paths[i], test_folder)
                save_path = os.path.join(wdsr_output_folder_final, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, single_output.astype('uint8'))
                wdsr_img = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
                original_img_path = os.path.join(bicubic_output_folder_final, relative_path)
                hr_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
                if hr_img is None:
                    raise FileNotFoundError(f"Cannot find original image at {original_img_path}")
                target_height, target_width = wdsr_img.shape
                hr_img = cv2.resize(hr_img, (target_width, target_height))
                psnr_value = psnr(hr_img, wdsr_img, data_range=255)
                ssim_value = ssim(hr_img, wdsr_img, data_range=255)
                results.append({'Filename': relative_path, 'PSNR': psnr_value, 'SSIM': ssim_value})
    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f"Results saved to {save_csv_path}")