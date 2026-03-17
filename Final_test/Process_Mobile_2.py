import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 设备配置
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
print(f"Using {device} device")

# 数据集类
class VideoFrameDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.imgs = []
        self.labels = []
        with open(self.file_path) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
            for img_path, label in samples:
                self.imgs.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return image, label

# 数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.RandomRotation(45),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# 加载数据集
train_dataset = VideoFrameDataset(file_path='./data/train2.txt', transform=data_transforms['train'])
valid_dataset = VideoFrameDataset(file_path='./data/test2.txt', transform=data_transforms['valid'])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

# 加载预训练的MobileNetV2模型并修改最后一层
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(device)


# 定义训练和测试函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def test(dataloader, model):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(torch.softmax(pred, dim=1).cpu().numpy())
            correct += (pred.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return np.array(all_labels), np.array(all_preds), avg_loss, accuracy

# 初始化训练参数以及记录变量
epochs = 50
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
# 训练和评估MobileNetV2并生成ROC曲线
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
    _, _, valid_loss, valid_accuracy = test(valid_dataloader, model)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

print('Evaluating MobileNetV2...')
labels, preds, valid_loss, valid_accuracy = test(valid_dataloader, model)

# 绘制多分类ROC曲线
plt.figure(figsize=(10, 8))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  # 三分类任务
    fpr[i], tpr[i], _ = roc_curve(labels == i, preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'MobileNetV2 class {i} (AUC = {roc_auc[i]:.2f})')

# 绘制随机分类器的参考线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for MobileNetV2')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 8))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# 绘制训练和验证的准确率曲线
plt.figure(figsize=(10, 8))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# 输出模型的准确率
print(f'MobileNetV2 Accuracy: {valid_accuracy * 100:.2f}%')