import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ====================
# 配置设备
# ====================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
print(f"Using {device} device")

# ====================
# 数据集类
# ====================
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

# ====================
# 数据增强
# ====================
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
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# ====================
# 加载预训练ResNet50模型并修改最后一层
# ====================
#model = models.resnet50(pretrained=True)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# ====================
# 定义训练和测试函数
# ====================
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

# ====================
# 曲线绘制函数
# ====================
def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ====================
# 初始化训练参数
# ====================
epochs = 50
patience = 5  # 早停阈值
best_val_loss = float('inf')
early_stop_counter = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

# ====================
# 训练循环
# ====================
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
    _, _, valid_loss, valid_accuracy = test(valid_dataloader, model)

    # 学习率调度器
    scheduler.step(valid_loss)

    # 记录损失和准确率
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

    # 早停机制
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered!")
        break

# ====================
# 绘制训练曲线
# ====================
plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

# ====================
# 最终评估
# ====================
print('Evaluating ResNet50...')
labels, preds, valid_loss, valid_accuracy = test(valid_dataloader, model)

# 绘制多分类ROC曲线
plt.figure(figsize=(10, 8))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  # 三分类任务
    fpr[i], tpr[i], _ = roc_curve(labels == i, preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'ResNet50 class {i} (AUC = {roc_auc[i]:.2f})')

# 绘制随机分类器的参考线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for ResNet50')
plt.legend(loc="lower right")
plt.grid()
plt.show()

print(f'ResNet50 Accuracy: {valid_accuracy * 100:.2f}%')
