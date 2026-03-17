import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# 定义数据增强
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

# 定义自定义数据集
class usp_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.imgs = []
        self.labels = []
        self.transform = transform
        with open(self.file_path) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
            for img_path, label in samples:
                self.imgs.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        image = Image.open(self.imgs[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        label = torch.tensor(label, dtype=torch.int64)
        return image, label

# 加载训练和测试数据
training_data = usp_dataset(file_path='./data/train2.txt', transform=data_transforms['train'])
testing_data = usp_dataset(file_path='./data/test2.txt', transform=data_transforms['valid'])
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

# 定义训练和测试函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model):
    model.eval()
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return np.array(all_labels), np.array(all_preds), accuracy

# 设备配置
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
print(f"Using {device} device")

# 模型列表
model_names = ['resnet50', 'resnet101', 'mobilenet_v2', 'vgg16']
models = {
    'resnet50': models.resnet50(pretrained=True),
    'resnet101': models.resnet101(pretrained=True),
    'mobilenet_v2': models.mobilenet_v2(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
}

# 修改最后一层以适应二分类任务
for name, model in models.items():
    if 'resnet' in name:
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'mobilenet' in name:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif 'vgg' in name:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    model = model.to(device)

accuracies = {}

# 训练和评估每个模型并生成ROC曲线
plt.figure(figsize=(10, 8))
for name, model in models.items():
    print(f'Training {name}...')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):  # 使用更少的epoch数目进行快速训练
        print(f'Epoch {epoch+1} for {name}')
        train(train_dataloader, model, loss_fn, optimizer)

    print(f'Evaluating {name}...')
    labels, preds, accuracy = test(test_dataloader, model)
    accuracies[name] = accuracy

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# 绘制随机分类器的参考线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 打印每种模型的准确率
for name, accuracy in accuracies.items():
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
