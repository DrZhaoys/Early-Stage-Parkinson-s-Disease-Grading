import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from Res_drop import training_data, train_dataloader
# 检查类别分布
labels = training_data.labels
counter = Counter(labels)
plt.bar(counter.keys(), counter.values())
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()

# 可视化部分训练数据
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_dataloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

# 简单模型评估
X_train = []
y_train = []
for img, label in train_dataloader:
    X_train.append(img.numpy())
    y_train.append(label.numpy())

X_train = np.array(X_train).reshape(len(X_train), -1)
y_train = np.array(y_train).ravel()

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))

# 交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-Validation Scores: ", scores)
print("Mean Score: ", scores.mean())
