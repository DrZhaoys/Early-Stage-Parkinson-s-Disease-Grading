import sys
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms, models
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
# 模型保存和加载
# ====================
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"Model path {path} does not exist.")

# ====================
# 定义训练和测试函数
# ====================
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for X, y in tqdm(dataloader, desc="Training"):
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

def test(dataloader, model, threshold=0.1):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
            all_labels.extend(y.cpu().numpy())
            pred_probs = torch.softmax(pred, dim=1)
            pred_classes = (pred_probs[:, 1] > threshold).long()  # Class 1 is predicted as positive
            all_preds.extend(torch.softmax(pred, dim=1).cpu().numpy())
            correct += (pred.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return np.array(all_labels), np.array(all_preds), avg_loss, accuracy

# ====================
# 计算与显示评估指标
# ====================
def calculate_metrics(labels, preds, num_classes):
    pred_classes = preds.argmax(axis=1)
    cm = confusion_matrix(labels, pred_classes)
    class_report = classification_report(labels, pred_classes, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
    print("Classification Report:")
    print(classification_report(labels, pred_classes, target_names=[f'Class {i}' for i in range(num_classes)]))
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [f'Class {i}' for i in range(num_classes)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
    return class_report

def plot_roc_curve(labels, preds, fold):
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):  # 几分类任务
        fpr[i], tpr[i], _ = roc_curve(labels == i, preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
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

def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve (Fold {fold})')
    plt.legend()
    plt.grid()
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

# ====================
# 主程序
# ====================
if __name__ == "__main__":
    mode = "test"  # 训练模式："train"，测试模式："test"
    model_path = "./saved_models/fold_1_model.pth"
    new_data_path = "./new_data/test_data.txt"  # 新数据集路径（测试模式）

    dataset = VideoFrameDataset(file_path='./data/train_final.txt', transform=data_transforms)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.array(dataset.labels)

    if mode == "train":
        os.makedirs('./saved_models', exist_ok=True)

        all_fold_metrics = []

        for fold, (train_idx, valid_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), 1):
            print(f'Fold {fold}')
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            valid_subset = torch.utils.data.Subset(dataset, valid_idx)

            train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
            valid_dataloader = DataLoader(valid_subset, batch_size=32, shuffle=False)

            # 修改为 EfficientNet-B0
            model = models.efficientnet_b7(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
            loss_fn = nn.CrossEntropyLoss()

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            # 创建日志文件
            log_file = open(f'./logs/final_efficient07_fold_{fold}_log.txt', 'w')
            log_file.write("Epoch\tTrain Loss\tTrain Acc\tValid Loss\tValid Acc\n")

            for epoch in range(50):  # 限制到50个epoch
                train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
                _, _, valid_loss, valid_accuracy = test(valid_dataloader, model)
                scheduler.step(valid_loss)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                train_accuracies.append(train_accuracy)
                valid_accuracies.append(valid_accuracy)

                # 打印每个 epoch 的指标
                print(f"Epoch {epoch + 1}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"  Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

                # 将指标写入日志文件
                log_file.write(
                    f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.4f}\t{valid_loss:.4f}\t{valid_accuracy:.4f}\n")

                # 关闭日志文件
            log_file.close()

            # 绘制训练曲线
            # plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold)

            # 验证集测试与结果计算
            fold_labels, fold_preds, _, fold_accuracy = test(valid_dataloader, model)
            # fold_auc = plot_roc_curve(fold_labels, fold_preds, fold)

            # 计算分类报告和混淆矩阵
            # fold_metrics = calculate_metrics(fold_labels, fold_preds, num_classes=4)

            # 存储折叠结果
            # all_fold_metrics.append({
            #     'accuracy': fold_accuracy,
            #     'auc': fold_auc,
            #     'loss': valid_losses[-1],
            #     'metrics': fold_metrics
            # })

            # 保存模型
            model_save_path = f'./saved_models/efficientnet_b7_{fold}_final.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model for Fold {fold} saved at {model_save_path}")

        # ====================
        # 综合评估
        # ====================
        overall_accuracy = np.mean([m['accuracy'] for m in all_fold_metrics])
        overall_auc = np.mean([np.mean(list(m['auc'].values())) for m in all_fold_metrics])
        overall_loss = np.mean([m['loss'] for m in all_fold_metrics])

        print(f"\nFinal Overall Results:")
        print(f"Accuracy: {overall_accuracy:.4f}")
        print(f"AUC: {overall_auc:.4f}")
        print(f"Loss: {overall_loss:.4f}")

        # 平均 F1-Score, Precision, Recall
        precision_scores = [np.mean([m['metrics'][f'Class {i}']['precision'] for i in range(3)]) for m in all_fold_metrics]
        recall_scores = [np.mean([m['metrics'][f'Class {i}']['recall'] for i in range(3)]) for m in all_fold_metrics]
        f1_scores = [np.mean([m['metrics'][f'Class {i}']['f1-score'] for i in range(3)]) for m in all_fold_metrics]

        print(f"Average Precision: {np.mean(precision_scores):.4f}")
        print(f"Average Recall: {np.mean(recall_scores):.4f}")
        print(f"Average F1-Score: {np.mean(f1_scores):.4f}")
    elif mode == "test":
        test_dataset = VideoFrameDataset(file_path='./data/test_final.txt', transform=data_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 修改为 EfficientNet-B0
        model = models.efficientnet_b7(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()

        # 遍历保存的模型并测试每个模型
        saved_models_dir = './saved_models/'
        if not os.path.exists(saved_models_dir):
            print("No saved models found.")
            sys.exit(1)  # 非零状态码表示异常退出

        all_test_results = []  # 存储每个模型的测试结果

        for fold in range(1, 6):  # 假设你有5折模型
            model_path = os.path.join(saved_models_dir, f'efficientnet_b7_{fold}_final.pth')
            if not os.path.exists(model_path):
                print(f"Model for Fold {fold} not found at {model_path}. Skipping...")
                continue

            print(f"Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # 测试模型
            test_labels, test_preds, test_loss, test_accuracy = test(test_dataloader, model)
            test_auc = plot_roc_curve(test_labels, test_preds, fold)
            test_metrics = calculate_metrics(test_labels, test_preds, num_classes=4)

            # 保存测试结果
            all_test_results.append({
                'fold': fold,
                'accuracy': test_accuracy,
                'auc': test_auc,
                'loss': test_loss,
                'metrics': test_metrics
            })

            print(f"Fold {fold} Test Results:")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  Loss: {test_loss:.4f}")
            # 打印 AUC 信息
            print(f"  AUCs:")
            for class_name, auc_value in test_auc.items():
                print(f"    {class_name}: {auc_value:.4f}")
            avg_auc = np.mean(list(test_auc.values()))
            print(f"  Average AUC: {avg_auc:.4f}")

        # 汇总所有折的测试结果
        if all_test_results:
            overall_test_accuracy = np.mean([r['accuracy'] for r in all_test_results])
            overall_test_auc = np.mean([np.mean(list(r['auc'].values())) for r in all_test_results])
            overall_test_loss = np.mean([r['loss'] for r in all_test_results])

            print("\nFinal Test Results:")
            print(f"  Overall Accuracy: {overall_test_accuracy:.4f}")
            print(f"  Overall AUC: {overall_test_auc:.4f}")
            print(f"  Overall Loss: {overall_test_loss:.4f}")
        else:
            print("No valid test results were obtained.")