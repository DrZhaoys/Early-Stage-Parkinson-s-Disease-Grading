import torch
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from WDSR_remake_ver import PreprocessDataset, DataLoader, device, transform, WDSR_SRResNet
# 定义模型，确保模型结构与保存的模型一致
model = WDSR_SRResNet().to(device)

# 加载保存的模型权重
model.load_state_dict(torch.load('wdsr_srresnet_model.pth'))
model.eval()  # 设置模型为评估模式

# 加载新数据集的文件夹
new_data_dir = 'E:/USP/test/frame_crop/'
new_dataset = PreprocessDataset(new_data_dir, transform=transform)
new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False)

# 开始计算 PSNR 和 SSIM
if __name__ == "__main__":
    # 开始计算 PSNR 和 SSIM
    psnr_ssim_results = []  # 用于存储每张图片的 PSNR 和 SSIM
    with torch.no_grad():
        for idx, (low_res, high_res) in enumerate(new_loader):  # 使用新的数据加载器
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            outputs = model(low_res)

            outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 numpy 格式
            high_res = high_res.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(outputs.shape[0]):
                # 计算每张图像的 PSNR 和 SSIM
                psnr_value = psnr(outputs[i], high_res[i], data_range=1.0)
                ssim_value = ssim(outputs[i], high_res[i], data_range=1.0, channel_axis=2)

                # 将图像索引、PSNR、SSIM 存入列表
                psnr_ssim_results.append({
                    "Image_Index": idx * new_loader.batch_size + i,  # 图像的索引
                    "PSNR": psnr_value,
                    "SSIM": ssim_value
                })

# 将结果保存到文件（如 CSV 文件）
df = pd.DataFrame(psnr_ssim_results)
df.to_csv("psnr_ssim_results_new_data.csv", index=False)
print("PSNR and SSIM for each image have been saved to 'psnr_ssim_results_new_data.csv'")
