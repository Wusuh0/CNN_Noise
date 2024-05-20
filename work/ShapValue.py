import shap
import torch
import os
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util.Model import Net,noiseNet,c1,c1_Mean,p1,p1_halfMean,p4,p4_halfMean
from util.Model_Func import test
from STL import STLPreprocess_Data

# 加载数据集
train_loader, test_loader = STLPreprocess_Data()
path = "model/stl/"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# for images, labels in train_loader:
#     images = images.to(device)
#     labels = labels.to(device)
#
# for images, labels in test_loader:
#     images = images.to(device)
#     labels = labels.to(device)

#加载类别
# class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
with open('data/stl10_binary/class_names.txt', 'r') as file:
    class_list = [line.strip() for line in file.readlines()]
class_names = np.array(class_list)
# 加载模型
model = Net()
model.load_state_dict(torch.load( "model/stl/epoch400.pth",map_location=lambda storage, loc: storage.cpu()))
model_noise = noiseNet()
model_noise.load_state_dict(torch.load( "model/stl/epoch400conv1_0.5mean.pth",map_location=lambda storage, loc: storage.cpu()))
print(model)
model.eval()

indices = torch.tensor([189])
test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[indices]
train_batch = next(iter(train_loader))
train_images, _ = train_batch
print(np.vectorize(lambda x: class_names[x])(labels[indices]))


# 创建Explainer实例

e = shap.GradientExplainer((model, model.features[0]), train_images)
e2 = shap.GradientExplainer((model_noise, model_noise.features[0]), train_images)
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=2)
shap_values2, indexes2 = e2.shap_values(to_explain, ranked_outputs=2)
print(np.array(shap_values).shape)
index_names = np.vectorize(lambda x: class_names[x])(indexes)
index_names2 = np.vectorize(lambda x: class_names[x])(indexes2)
new_to_explain = np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
shap_values2 = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values2]
print(np.array(shap_values).shape)
print(np.array(shap_values2).shape)
npPath = "result/stl/Shap_value/dog20_ship112/"
if not os.path.exists(npPath):
    os.makedirs(npPath)
np.save(npPath + "f0.npy", np.array(shap_values))
np.save(npPath + "conv1_0.5mean_f0.npy", np.array(shap_values2))
# 统计shap值的sum
shap1_sum = []
shap2_sum = []
for i in range(len(shap_values)):
    temp = []
    for j in range(len(shap_values[i])):
        temp.append(np.sum(shap_values[i][j]))
    shap1_sum.append(temp)

for i in range(len(shap_values2)):
    temp = []
    for j in range(len(shap_values2[i])):
        temp.append(np.sum(shap_values2[i][j]))
    shap2_sum.append(temp)
print(shap1_sum)
print(shap2_sum)
sum = [shap1_sum, shap2_sum]
with open(npPath + 'sum.txt', 'a') as file:
    file.write("no_f0 conv1_0.5mean_F0\n")
str_list = [' '.join(map(str, item)) for item in sum]
with open(npPath + 'sum.txt', 'a') as file:
    for item in str_list:
        file.write(item + '\n')
# 交换shap矩阵，对比两个模型在相同图片上的shap值
# new_to_explain1 = new_to_explain.copy()
# new_to_explain1[1] = new_to_explain1[0].copy()
# new_to_explain2 = new_to_explain.copy()
# new_to_explain2[0] = new_to_explain2[1].copy()
#
# shap_values1 = shap_values.copy()
# temp = shap_values[0][1].copy()
# shap_values1[0][1] = shap_values2[0][0].copy()
# shap_values2[0][0] = temp
#
# index_names1 = index_names.copy()
# temp = index_names1[1].copy()
# index_names1[1] = index_names2[0].copy()
# index_names2[0] = temp


# 可视化shap_value
save_path = 'result/stl/new/airplane car/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

shap.image_plot(shap_values, new_to_explain, index_names)
# plt.savefig(save_path + 'f'+ str(i) +'.png')
# plt.close()
shap.image_plot(shap_values2, new_to_explain, index_names2)
# plt.savefig(save_path + 'pool4_0.5mean_beF'+ str(i+1) +'.png')
# plt.close()
# show = False




