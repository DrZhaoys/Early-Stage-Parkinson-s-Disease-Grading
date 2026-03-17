import sys
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor

# ====================
# Configure Device
# ====================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# ====================
# Set Random Seed
# ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set random seed to 42, can be modified as needed

# ====================
# Dataset Class
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
        # 使用处理器将图像转换为模型输入格式
        inputs = processor(images=image, return_tensors="pt")
        image = inputs['pixel_values'].squeeze(0)  # 移除 batch 维度
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return image, label


import os
os.environ["HF_HUB_OFFLINE"] = "1"
# ====================
# Data Augmentation (Adjusted for ViT)
# ====================
# 初始化图像处理器
#processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', mirror='https://hf-mirror.com')
# 加载本地模型和预处理器
try:
    processor = ViTImageProcessor.from_pretrained(
        "E:/deeplearning/vit-base-patch16-224-in21k",
        local_files_only=True,
        do_rescale=False  # 避免重复缩放
    )
    model = ViTForImageClassification.from_pretrained("E:/deeplearning/vit-base-patch16-224-in21k", num_labels=4)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model = model.to(device)
print("Model loaded successfully!")
# 更新数据变换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 期望 224x224 输入
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # 注意：ViTImageProcessor 会处理归一化，因此这里不需要手动 Normalize
])

# 更新 Dataset 类中的 __getitem__ 方法

# ====================
# Model Saving and Loading
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
# Training and Testing Functions
# ====================
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        logits = pred.logits  # 提取 logits
        loss = loss_fn(logits, y)  # 使用 logits 计算损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        predicted = logits.argmax(1)  # 使用 logits 进行预测
        correct += (predicted == y).sum().item()  # 计算正确预测数量
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
            logits = pred.logits  # 提取 logits
            loss = loss_fn(logits, y)  # 使用 logits 计算损失
            total_loss += loss.item() * X.size(0)
            all_labels.extend(y.cpu().numpy())
            pred_probs = torch.softmax(logits, dim=1)
            pred_classes = (pred_probs[:, 1] > threshold).long()  # Class 1 is predicted as positive
            all_preds.extend(pred_probs.cpu().numpy())
            predicted = logits.argmax(1)  # 使用 logits 进行预测
            correct += (predicted == y).sum().item()  # 计算正确预测数量
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return np.array(all_labels), np.array(all_preds), avg_loss, accuracy

# ====================
# Calculate and Display Evaluation Metrics
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
    for i in range(4):  # 4-class task
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
# Main Program
# ====================
if __name__ == "__main__":
    mode = "test"  # Training mode: "train", Testing mode: "test"

    dataset = VideoFrameDataset(file_path='./data/train_final.txt', transform=data_transforms)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.array(dataset.labels)

    if mode == "train":
        os.makedirs('./saved_models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)  # Create logs directory

        all_fold_metrics = []
        # Initialize Vision Transformer model
        # 初始化 Vision Transformer 模型
        model = ViTForImageClassification.from_pretrained(
            "E:/deeplearning/vit-base-patch16-224-in21k",
            num_labels=4,
            local_files_only=True
        )
        model = model.to(device)
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels), 1):
            print(f'Fold {fold}')
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            valid_subset = torch.utils.data.Subset(dataset, valid_idx)

            train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
            valid_dataloader = DataLoader(valid_subset, batch_size=32, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
            # 在训练循环中
            print(f"Learning rate: {scheduler.get_last_lr()}")
            loss_fn = nn.CrossEntropyLoss()

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            # Create log file
            log_file = open(f'./logs/final_vit_16_fold_{fold}_log.txt', 'w')
            log_file.write("Epoch\tTrain Loss\tTrain Acc\tValid Loss\tValid Acc\n")

            for epoch in range(50):  # Limit to 50 epochs
                train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
                _, _, valid_loss, valid_accuracy = test(valid_dataloader, model)
                scheduler.step(valid_loss)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                train_accuracies.append(train_accuracy)
                valid_accuracies.append(valid_accuracy)

                # Print metrics for each epoch
                print(f"Epoch {epoch+1}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"  Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

                # Write metrics to log file
                log_file.write(f"{epoch+1}\t{train_loss:.4f}\t{train_accuracy:.4f}\t{valid_loss:.4f}\t{valid_accuracy:.4f}\n")

            # Close log file
            log_file.close()

            # Plot training curves (optional, currently commented)
            # plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, fold)

            # Validation set testing and results calculation
            fold_labels, fold_preds, _, fold_accuracy = test(valid_dataloader, model)
            # fold_auc = plot_roc_curve(fold_labels, fold_preds, fold)

            # Calculate classification report and confusion matrix (optional, currently commented)
            # fold_metrics = calculate_metrics(fold_labels, fold_preds, num_classes=4)

            # Store fold results (optional, currently commented)
            # all_fold_metrics.append({
            #     'accuracy': fold_accuracy,
            #     'auc': fold_auc,
            #     'loss': valid_losses[-1],
            #     'metrics': fold_metrics
            # })

            # Save model
            model_save_path = f'./saved_models/final_vit_16_{fold}_final.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model for Fold {fold} saved at {model_save_path}")

        # ====================
        # Overall Evaluation (optional, currently commented)
        # ====================
        # overall_accuracy = np.mean([m['accuracy'] for m in all_fold_metrics])
        # overall_auc = np.mean([np.mean(list(m['auc'].values())) for m in all_fold_metrics])
        # overall_loss = np.mean([m['loss'] for m in all_fold_metrics])
        # print(f"\nFinal Overall Results:")
        # print(f"Accuracy: {overall_accuracy:.4f}")
        # print(f"AUC: {overall_auc:.4f}")
        # print(f"Loss: {overall_loss:.4f}")

    elif mode == "test":
        test_dataset = VideoFrameDataset(file_path='./data/test_final.txt', transform=data_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load model structure
        model = ViTForImageClassification.from_pretrained(
            "E:/deeplearning/vit-base-patch16-224-in21k",
            num_labels=4,
            local_files_only=True
        )
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()

        saved_models_dir = './saved_models/'
        if not os.path.exists(saved_models_dir):
            print("No saved models found.")
            sys.exit(1)

        all_test_results = []
        for fold in range(1, 6):
            model_path = os.path.join(saved_models_dir, f'final_vit_16_{fold}_final.pth')
            if not os.path.exists(model_path):
                print(f"Model for Fold {fold} not found at {model_path}. Skipping...")
                continue

            print(f"Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path))
            model.eval()

            test_labels, test_preds, test_loss, test_accuracy = test(test_dataloader, model)
            test_auc = plot_roc_curve(test_labels, test_preds, fold)
            test_metrics = calculate_metrics(test_labels, test_preds, num_classes=4)

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
            print(f"  AUCs:")
            for class_name, auc_value in test_auc.items():
                print(f"    {class_name}: {auc_value:.4f}")
            avg_auc = np.mean(list(test_auc.values()))
            print(f"  Average AUC: {avg_auc:.4f}")

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