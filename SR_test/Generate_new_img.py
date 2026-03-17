import torch
from torchvision import transforms
from PIL import Image
import os
from WDSR_Validation_alone import WDSR_SRResNet, device
# 定义数据预处理，确保与训练时一致
transform = transforms.Compose([transforms.ToTensor()])

# 加载保存的模型
model = WDSR_SRResNet().to(device)
model.load_state_dict(torch.load('wdsr_srresnet_model.pth'))
model.eval()  # 设置为评估模式


# 定义生成高分辨率图像的函数
def generate_and_save_sr_images(input_dir, output_dir, model, transform): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入图像目录
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path)

        # 预处理低分辨率图像
        img_lr = transform(img).unsqueeze(0).to(device)  # 添加批次维度

        # 使用模型生成高分辨率图像
        with torch.no_grad():
            sr_img = model(img_lr)

        # 将 Tensor 转换为 PIL 图像
        sr_img = sr_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 去除批次维度并转换为 (H, W, C)
        sr_img = (sr_img * 255.0).clip(0, 255).astype('uint8')  # 转换为 0-255 的 uint8 格式
        sr_img = Image.fromarray(sr_img)

        # 保存生成的高分辨率图像
        output_path = os.path.join(output_dir, f'sr_{img_name}')
        sr_img.save(output_path)

        print(f'Saved: {output_path}')


# 定义输入和输出目录
input_dir = 'E:/USP/1/'  # 低分辨率图像目录
output_dir = 'E:/USP/output_sr_images/'   # 输出高分辨率图像的保存目录

# 生成并保存高分辨率图像
generate_and_save_sr_images(input_dir, output_dir, model, transform)
