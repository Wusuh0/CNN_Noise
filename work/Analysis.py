import shap
import torch
import os
import numpy as np
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util.Model_Func import test
from Cifar10 import Preprocess_Data,VGG,noiseVGG

path = "model/cifar10/"
train_loader, test_loader = Preprocess_Data()

#类名
class_names = ['airplane','Automobile', 'bird' , 'cat' ,'deer' ,'dog', 'frog', 'horse' ,'ship' ,'truck']
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
Loc = [1,3,7,14]
for loc in Loc:
    count = []
    percentage = []
    print(f"layer{loc} vs no")
    for i in range(1, 6):
        model = VGG('VGG16', 10)
        model.load_state_dict(
            torch.load(path + f"no/epoch{i * 20}.pth", map_location=lambda storage, loc: storage.cpu()))
        model.eval()
        print(f"epoch{i * 20}")
        model_noise = noiseVGG('VGG16', 10, loc)
        model_noise.load_state_dict(
            torch.load(path + f"layer{loc}/epoch{i * 20}.pth", map_location=lambda storage, loc: storage.cpu()))
        model_noise.eval()

        total = 0
        # 两模型预测相同的数量
        equalCnt = 0
        # 都预测对的数量
        accCnt = 0
        # 都预测错的数量
        errorCnt = 0
        # 加噪模型预测对而无噪错误
        noiseCnt = 0
        # 相反
        Cnt = 0

        with torch.no_grad():  # 禁用梯度计算以节省内存和计算时间
            for data, labels in test_loader:
                res = model(data)
                resn = model_noise(data)
                total += labels.size(0)
                for j in range(len(data)):
                    # 预测相同
                    if res[j].argmax() == resn[j].argmax():
                        equalCnt = equalCnt + 1
                        # 预测正确
                        if res[j].argmax() == labels[j]:
                            accCnt = accCnt + 1
                        # 预测错误
                        else:
                            errorCnt = errorCnt + 1
                    # 预测不同
                    else:
                        # 无噪模型正确
                        if res[j].argmax() == labels[j]:
                            Cnt = Cnt + 1
                        # 加噪模型正确
                        elif resn[j].argmax() == labels[j]:
                            noiseCnt = noiseCnt + 1

        print(equalCnt, accCnt, errorCnt)
        print(Cnt, noiseCnt)
        count.append(np.array([equalCnt, accCnt, errorCnt, Cnt, noiseCnt]))

        print(f"{equalCnt / total * 100}%")
        print(f"{accCnt / total * 100}%")
        print(f"{errorCnt / total * 100}%")
        print(f"{Cnt / total * 100}%")
        print(f"{noiseCnt / total * 100}%")
        percentage.append(np.array(
            [equalCnt / total * 100, accCnt / total * 100, errorCnt / total * 100, Cnt / total * 100,
             noiseCnt / total * 100]))

    np.savez(path + f"layer{loc}/count.npz", epoch20=count[0], epoch40=count[1], epoch60=count[2], epoch80=count[3],
             epoch100=count[4])
    np.savez(path + f"layer{loc}/percentage.npz", epoch20=percentage[0], epoch40=percentage[1], epoch60=percentage[2],
             epoch80=percentage[3], epoch100=percentage[4])

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

