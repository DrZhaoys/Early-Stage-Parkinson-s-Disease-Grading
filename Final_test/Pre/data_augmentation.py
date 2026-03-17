import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
# 定义数据增强变换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomRotation(45),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class usp_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        # 接受两参数，file_path 和 transform，file_path 是包含图像文件路径和标签的文本文件路径，transform 是数据预处理的转换。
        self.file_path = file_path
        self.imgs = []
        self.labels = []
        self.transform = transform
        # 初始化了数据集的实例变量 self.file_path，self.imgs，self.labels 和 self.transform。
        with open(self.file_path) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
            # 打开指定的文本文件 file_path，并逐行读取文件内容。每行应包含图像路径和对应标签，以空格分隔。
            for img_path, label in samples:
                self.imgs.append(img_path)
                self.labels.append(label)
            # 将图像路径和标签分别添加到 self.imgs 和 self.labels 列表中，以准备后续使用。

    def __len__(self):
        return len(self.imgs)
        # 该方法返回数据集的长度，即数据集中图像的数量。

    def __getitem__(self, item):
        # 接受一个整数 item，表示数据集中的一个样本索引。
        image = Image.open(self.imgs[item])
        # 打开图像文件 self.imgs[item] 使用PIL库，并将其存储在变量 image 中。
        if self.transform:
            image = self.transform(image)
            # 如果定义了数据预处理转换 self.transform，则将图像应用这些转换，以便在模型训练过程中使用。
        label = self.labels[item]
        # 从 self.labels 中获取相应索引的标签，并将其存储在变量 label 中。
        label = torch.from_numpy(np.array(label, dtype=np.int64))
        # 转换标签为PyTorch的Tensor对象，使用torch.from_numpy(np.array(label, dtype=np.int64))。
        return image, label


training_data = usp_dataset(file_path='../data/train.txt', transform=data_transforms['train'])
testing_data = usp_dataset(file_path='../data/test.txt', transform=data_transforms['valid'])

train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=8, shuffle=True)


# 获取一批增强后的图像
data_iter = iter(train_dataloader)
images, _ = data_iter.next()

# 定义反标准化变换
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# 反标准化图像
images = inv_normalize(images)

# 可视化图像
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for idx in range(4):
    ax = axes[idx]
    img = images[idx].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis('off')
plt.show()
