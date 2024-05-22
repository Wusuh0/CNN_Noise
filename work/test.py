import torch
import numpy as np
import os
from torch import nn
from STL import Net,STLPreprocess_Data
from util.Model import noiseNet,c1_Mean,p1_halfMean,p4_halfMean,c5_halfMean
from util.Model_Func import test,preprocess_Data
#['airplane' 'bird' 'car' 'cat' 'deer' 'dog' 'horse' 'monkey' 'ship' 'truck']

# 读取数据
train_loader, test_loader = STLPreprocess_Data()
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
model = Net()
model.load_state_dict(torch.load("model/stl/grad_0.5mean/new_epoch40.pth",map_location=lambda storage, loc: storage.cpu()))
model.eval()
test(model, device, test_loader, criterion)
# for i in range(1,5):
#     model.load_state_dict(torch.load(f"model/stl/conv1_mean/epoch{i*100}conv1_mean.pth",
#                                      map_location=lambda storage, loc: storage.cpu()))
#     model.eval()
#     print(i*100)
#     num_classes = 10
#     # 使用混淆矩阵计算TP, FP, TN, FN
#     confusion_matrix = torch.zeros((num_classes, num_classes))
#     with torch.no_grad():  # 禁用梯度计算以节省内存和计算时间
#         for data, labels in test_loader:
#             predicts = model(data)
#             for i in range(len(labels)):
#                 confusion_matrix[labels[i], predicts[i].argmax()] += 1
#     print(confusion_matrix.sum())
#     print(confusion_matrix)
#
#     recalls = []
#     precisions = []
#     for i in range(num_classes):
#         TP = confusion_matrix[i, i]
#         FN = torch.sum(confusion_matrix[i, :]) - TP
#         FP = torch.sum(confusion_matrix[:, i]) - TP
#         recall = TP / (TP + FN)
#         if (FP + TP) == 0:
#             precision = torch.tensor(0)
#         else:
#             precision = TP / (TP + FP)
#         recalls.append(recall)
#         precisions.append(precision)
#
#     # 计算平均召回率
#     average_recall = torch.mean(torch.tensor(recalls))
#     average_precision = torch.mean(torch.tensor(precisions))
#
#     print("all classes recall:", [round(t.item(), 2) for t in recalls])
#     print("avg recall:", round(average_recall.item(), 2))
#
#     print("all classes precision:", [round(t.item(), 2) for t in precisions])
#     print("avg precision:", round(average_precision.item(), 2))

