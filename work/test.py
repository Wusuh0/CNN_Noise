import torch
import numpy as np
import os
from torch import nn
from Cifar10 import VGG,noiseVGG, Preprocess_Data
from util.Model_Func import test
#['airplane' 'bird' 'car' 'cat' 'deer' 'dog' 'horse' 'monkey' 'ship' 'truck'] STL
#['airplane' 'Automobile' 'bird'  'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'] Cifar10
# 读取数据
train_loader, test_loader = Preprocess_Data()
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
Loc = [1,3,7,14]
for loc in Loc:
    print(f"layer{loc}")
    for i in range(1, 6):
        model = noiseVGG('VGG16', 10, loc)
        model.load_state_dict(
            torch.load(f"model/cifar10/layer{loc}/epoch{i * 20}.pth", map_location=lambda storage, loc: storage.cpu()))
        model.eval()
        test(model, device, test_loader, criterion)


Loc = [1,3,7,14]
for loc in Loc:
    print(f"layer{loc}")
    avg_recall = []
    avg_precision = []
    for i in range(1, 6):
        model = noiseVGG('VGG16', 10, loc)
        model.load_state_dict(
            torch.load(f"model/cifar10/layer{loc}/epoch{i * 20}.pth", map_location=lambda storage, loc: storage.cpu()))
        model.eval()
        print(f"epoch{i * 20}")
        num_classes = 10
        # 使用混淆矩阵计算TP, FP, TN, FN
        confusion_matrix = torch.zeros((num_classes, num_classes))
        with torch.no_grad():  # 禁用梯度计算以节省内存和计算时间
            for data, labels in test_loader:
                predicts = model(data)
                for i in range(len(labels)):
                    confusion_matrix[labels[i], predicts[i].argmax()] += 1
        print(confusion_matrix.sum())
        print(confusion_matrix)

        recalls = []
        precisions = []
        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FN = torch.sum(confusion_matrix[i, :]) - TP
            FP = torch.sum(confusion_matrix[:, i]) - TP
            recall = TP / (TP + FN)
            if (FP + TP) == 0:
                precision = torch.tensor(0)
            else:
                precision = TP / (TP + FP)
            recalls.append(recall)
            precisions.append(precision)

        # 计算平均召回率
        average_recall = torch.mean(torch.tensor(recalls))
        average_precision = torch.mean(torch.tensor(precisions))
        avg_recall.append(average_recall)
        avg_precision.append(average_precision)


