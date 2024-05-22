import shap
import torch
import os
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util.Model import Net,noiseNet,c1_Mean,c5_halfMean,p1_halfMean,p4_halfMean
from util.Model_Func import test
from STL import STLPreprocess_Data

path = "model/stl/"
train_loader, test_loader = STLPreprocess_Data()

#类名
with open('data/stl10_binary/class_names.txt', 'r') as file:
    class_list = [line.strip() for line in file.readlines()]
class_names = np.array(class_list)
print(class_names)
model = Net()
model.load_state_dict(torch.load( path + "epoch400.pth",map_location=lambda storage, loc: storage.cpu()))
model_noise = c1_Mean()
model_noise.load_state_dict(torch.load(path +  "conv1_0.25mean/epoch400conv1_0.25mean.pth",map_location=lambda storage, loc: storage.cpu()))
model.eval()
model_noise.eval()

# indices = torch.tensor([24])
# test_batch = next(iter(test_loader))
# test_images, labels = test_batch
# res = np.load("result/res.npy")
# resn = np.load("result/resn.npy")
# print(res[24])
# print(resn[24])
# with torch.no_grad():
#     res = model(test_images)
#     np.save("result/res.npy",res.numpy())
#     resn = model_noise(test_images)
#     np.save("result/resn.npy",resn.numpy())
#     print(res[indices])
#     print(resn[indices])

total = 0

#两模型预测相同的数量
equalCnt = 0
#都预测对的数量
accCnt = 0
#都预测错的数量
errorCnt = 0
#加噪模型预测对而无噪错误
noiseCnt = 0
#相反
Cnt = 0

with torch.no_grad():  # 禁用梯度计算以节省内存和计算时间
    for data, labels in test_loader:
        res = model(data)
        resn = model_noise(data)
        print(res.shape)
        total += labels.size(0)
        for i in range(len(data)):
            # 预测相同
            if res[i].argmax() == resn[i].argmax():
                equalCnt = equalCnt + 1
                # 预测正确
                if res[i].argmax() == labels[i]:
                    accCnt = accCnt + 1
                # 预测错误
                else:
                    errorCnt = errorCnt + 1
            # 预测不同
            else:
                # 无噪模型正确
                if res[i].argmax() == labels[i]:
                    Cnt = Cnt + 1
                # 加噪模型正确
                elif resn[i].argmax() == labels[i]:
                    noiseCnt = noiseCnt + 1



print(equalCnt,accCnt,errorCnt)
print(Cnt,noiseCnt)

print(f"{equalCnt/total*100}%")
print(f"{accCnt/total*100}%")
print(f"{errorCnt/total*100}%")

print(f"{Cnt/total*100}%")
print(f"{noiseCnt/total*100}%")