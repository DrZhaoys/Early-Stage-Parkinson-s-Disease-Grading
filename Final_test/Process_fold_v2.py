import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================
# 配置设备
# ====================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
data_transforms = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.RandomRotation(45),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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
def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold):
    epochs = range(1, len(train_losses) + 1)
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve (Fold {fold})')
    plt.legend()
    plt.grid()
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve (Fold {fold})')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_roc_curve(labels, preds, fold):
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 三分类任务
        fpr[i], tpr[i], _ = roc_curve(labels == i, preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # 绘制随机分类器的参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Fold {fold})')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    return roc_auc


# ====================
# 交叉验证
# ====================
dataset = VideoFrameDataset(file_path='./data/all_data.txt', transform=data_transforms)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels = np.array(dataset.labels)

all_fold_metrics = []

for fold, (train_idx, valid_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), 1):
    print(f'Fold {fold}')
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    valid_subset = torch.utils.data.Subset(dataset, valid_idx)

    train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_subset, batch_size=64, shuffle=False)

    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(20):  # 限制到20个epoch
        train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
        _, _, valid_loss, valid_accuracy = test(valid_dataloader, model)
        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

    plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold)
    fold_labels, fold_preds, _, fold_accuracy = test(valid_dataloader, model)
    fold_auc = plot_roc_curve(fold_labels, fold_preds, fold)

    fold_metrics = {
        'accuracy': fold_accuracy,
        'auc': fold_auc,
        'loss': valid_losses[-1]
    }
    all_fold_metrics.append(fold_metrics)

# ====================
# 整体评估
# ====================
overall_accuracy = np.mean([m['accuracy'] for m in all_fold_metrics])
overall_auc = np.mean([np.mean(list(m['auc'].values())) for m in all_fold_metrics])
overall_loss = np.mean([m['loss'] for m in all_fold_metrics])

print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Overall AUC: {overall_auc:.4f}")
print(f"Overall Loss: {overall_loss:.4f}")
