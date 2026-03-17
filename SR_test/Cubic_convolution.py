import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# 路径设置
high_res_folder = 'E:/USP/test/frame_crop/'
output_csv = 'E:/USP/test/pytorch_bicubic_results_2x.csv'

# 保存结果
results = []

# 将 NumPy 图像转换为 PyTorch 张量
def numpy_to_tensor(image):
    # 转为 float 类型并归一化
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    return image

# 将 PyTorch 张量转换为 NumPy 图像
def tensor_to_numpy(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()  # (1, C, H, W) -> (H, W, C)
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)  # 反归一化
    return tensor

# 遍历高分辨率图像
for filename in tqdm(os.listdir(high_res_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取高分辨率图像
        high_res = cv2.imread(os.path.join(high_res_folder, filename))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 获取图像尺寸
        h, w, c = high_res.shape

        # 下采样
        low_res = cv2.resize(high_res, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        # 转换为 PyTorch 张量
        low_res_tensor = numpy_to_tensor(low_res)

        # 使用 PyTorch 进行立方卷积上采样
        upscaled_tensor = F.interpolate(low_res_tensor, size=(h, w), mode='bicubic', align_corners=False)

        # 转换回 NumPy 格式
        upscaled = tensor_to_numpy(upscaled_tensor)

        # 计算 PSNR 和 SSIM
        psnr_value = psnr(high_res, upscaled)
        ssim_value = ssim(high_res, upscaled, channel_axis=2)

        # 记录结果
        results.append({
            'Filename': filename,
            'PSNR': psnr_value,
            'SSIM': ssim_value
        })

# 保存到 CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f'Results saved to {output_csv}')
