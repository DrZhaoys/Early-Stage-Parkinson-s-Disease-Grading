import os
from PIL import Image

def downsample_image(image_path, output_dir, factor):
    """下采样图像并保存到指定目录"""
    with Image.open(image_path) as img:
        # 计算新尺寸（确保至少1x1像素）
        new_size = (
            max(1, img.width // factor),
            max(1, img.height // factor)
        )

        # 使用BICUBIC（双立方插值法）下采样
        resized_img = img.resize(new_size, Image.Resampling.BICUBIC)

        # 构造输出路径（保持文件名，改为PNG格式）
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}.png")
        resized_img.save(output_path, "PNG")


def process_dataset(input_dir, output_root, factors):
    """处理整个数据集"""
    # 获取所有类别目录
    classes = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d))]

    for factor in factors:
        print(f"Processing factor x{factor}...")
        factor_dir = os.path.join(output_root, f"x{factor}")

        for class_name in classes:
            # 创建输出目录
            class_input = os.path.join(input_dir, class_name)
            class_output = os.path.join(factor_dir, class_name)
            os.makedirs(class_output, exist_ok=True)

            # 处理每个图像
            for img_name in os.listdir(class_input):
                img_path = os.path.join(class_input, img_name)
                if os.path.isfile(img_path):
                    try:
                        downsample_image(img_path, class_output, factor)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")


if __name__ == "__main__":
    # 直接在代码中定义输入和输出路径
    input_dir = "E:/USPPP/"  # 输入数据集路径
    output_root = "E:/DownSample/"  # 输出根目录路径

    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    # 执行下采样处理
    process_dataset(input_dir, output_root, [2, 4, 8])

    print("所有下采样操作已完成！输出目录结构：")
    print(f"""
{output_root}
├── x2
│   ├── 0
│   ├── 1
│   ├── 2
│   └── 3
├── x4
└── x8
    """)
