import shap
import torch
import os
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from STL import Net,noiseNet
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
model.load_state_dict(torch.load( "model/stl/epoch100.pth",map_location=lambda storage, loc: storage.cpu()))
model_noise = noiseNet()
model_noise.load_state_dict(torch.load( "model/stl/epoch100conv1_0.5mean.pth",map_location=lambda storage, loc: storage.cpu()))
print(model)
model.eval()

indices = torch.tensor([13,24])
test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[indices]
train_batch = next(iter(train_loader))
train_images, _ = train_batch
print(np.vectorize(lambda x: class_names[x])(labels[indices]))

# res = model(test_images)
# res = res.detach().numpy()
# resn = model_noise(test_images)
# resn = resn.detach().numpy()
# for i in range(200):
#     if res[i].argmax() == resn[i].argmax() and res[i].argmax() == labels[i]:
#         print(i)


# 创建Explainer实例
for i in range(0,11):
    e = shap.GradientExplainer((model, model.features[i]), train_images)
    e2 = shap.GradientExplainer((model_noise, model_noise.features[i]), train_images)
    shap_values, indexes = e.shap_values(to_explain, ranked_outputs=2)
    shap_values2, indexes2 = e2.shap_values(to_explain, ranked_outputs=2)
    print(np.array(shap_values).shape)
    index_names = np.vectorize(lambda x: class_names[x])(indexes)
    index_names2 = np.vectorize(lambda x: class_names[x])(indexes2)
    new_to_explain = np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
    shap_values2 = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values2]
    #
    # new_to_explain1 = new_to_explain
    # new_to_explain1[1] = new_to_explain1[0]
    #
    # new_to_explain2 = new_to_explain
    # new_to_explain2[0] = new_to_explain2[1]
    #
    # temp = shap_values[1]
    # shap_values[1] = shap_values2[0]
    # shap_values2[0] = temp
    #
    # temp2 = index_names[1]
    # index_names[1] = index_names2[0]
    # index_names2[0] = temp2
    # 可视化shap_value
    save_path = 'result/stl/epoch100/airplane car/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shap.image_plot(shap_values, new_to_explain, index_names,show = False)
    plt.savefig(save_path + 'f'+ str(i) +'.png')
    plt.close()
    shap.image_plot(shap_values2, new_to_explain, index_names2,show = False)
    plt.savefig(save_path + 'conv1_0.5mean_f'+ str(i) +'.png')
    plt.close()
    # show = False




