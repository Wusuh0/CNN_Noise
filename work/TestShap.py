import shap
import torch
import os
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from MNIST import Net,noiseNet
from util.Model_Func import test
from MNIST import MNISTPreprocess_Data

# 加载数据集
train_loader, test_loader = MNISTPreprocess_Data()
path = "model/mnist/"

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
# with open('data/stl10_binary/class_names.txt', 'r') as file:
#     class_list = [line.strip() for line in file.readlines()]
# class_names = np.array(class_list)
# 加载模型
model = noiseNet()
model.load_state_dict(torch.load( path + "epoch5_conv1_10.pth",map_location=lambda storage, loc: storage.cpu()))
model_noise = noiseNet()
model_noise.load_state_dict(torch.load( path + "epoch5_conv1_20.pth",map_location=lambda storage, loc: storage.cpu()))
print(model)
model.eval()

test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[30:32]
train_batch = next(iter(train_loader))
train_images, _ = train_batch

# 创建Explainer实例
e = shap.GradientExplainer((model, model.conv1), train_images)
e2 = shap.GradientExplainer((model_noise, model_noise.conv1), train_images)
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1)
shap_values2, indexes2 = e2.shap_values(to_explain, ranked_outputs=1)
print(np.array(shap_values).shape)

shap1_sum = []
shap2_sum = []
print(len(shap_values[0]))
for i in range(len(shap_values[0])):
    shap1_sum.append(np.sum(shap_values[0][i]))
for i in range(len(shap_values[0])):
    shap2_sum.append(np.sum(shap_values2[0][i]))

print(np.sum(shap_values[0][0]))
print(np.sum(shap_values2[0][0]))

print(np.sum(shap_values[0][1]))
print(np.sum(shap_values2[0][1]))

index_names = np.vectorize(lambda x: str(x))(indexes)
index_names2 = np.vectorize(lambda x: str(x))(indexes2)
print(index_names)
print(index_names2)

new_to_explain = np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
shap_values2 = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values2]

new_to_explain1 = new_to_explain.copy()
new_to_explain1[1] = new_to_explain1[0].copy()

new_to_explain2 = new_to_explain.copy()
new_to_explain2[0] = new_to_explain2[1].copy()

shap_values1 = shap_values.copy()
temp = shap_values[0][1].copy()
shap_values1[0][1] = shap_values2[0][0].copy()
shap_values2[0][0] = temp

index_names1 = index_names.copy()
temp = index_names1[1].copy()
index_names1[1] = index_names2[0].copy()
index_names2[0] = temp

# 可视化shap_value
save_path1 = 'result/mnist/epoch5/10/'
if not os.path.exists(save_path1):
    os.makedirs(save_path1)
save_path2 = 'result/mnist/epoch5/20/'
if not os.path.exists(save_path2):
    os.makedirs(save_path2)

shap.image_plot(shap_values, new_to_explain1, index_names)
# plt.savefig(save_path1 + '10_conv1.png')
# plt.close()
shap.image_plot(shap_values2, new_to_explain2, index_names2)
# plt.savefig(save_path2 + '20_conv1.png')
# plt.close()
# show = False




