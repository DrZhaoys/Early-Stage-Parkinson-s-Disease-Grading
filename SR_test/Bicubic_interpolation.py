import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# 路径设置
high_res_folder = 'E:/USP/test/test_data/'
output_csv = 'E:/USP/test/bicubic_article_pic_only.csv'
output_image_folder = 'E:/USP/test/bicubic_8x/'  # 插值后图像的输出文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_image_folder, exist_ok=True)
# 保存结果
results = []

# 遍历高分辨率图像
for filename in tqdm(os.listdir(high_res_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取高分辨率图像
        high_res = cv2.imread(os.path.join(high_res_folder, filename))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 获取图像尺寸
        h, w, c = high_res.shape

        # 下采样
        low_res = cv2.resize(high_res, (w // 8, h // 8), interpolation=cv2.INTER_AREA) #倍数自己改

        # 使用双三次插值放大回原始分辨率
        upscaled = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)

        # 保存插值后的图像
        output_image_path = os.path.join(output_image_folder, filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))

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
