import shap
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from MNIST import Net,noiseNet
from util.Model_Func import preprocess_Data
from MNIST import MNISTPreprocess_Data
# 加载数据集
train_loader, test_loader = MNISTPreprocess_Data()
path = "model/mnist/"

#加载类别

# 加载模型
model = Net()
model.load_state_dict(torch.load(path + "epoch5_pool1_1.pth"))
model.eval()

test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[:3]
train_batch = next(iter(train_loader))
train_images, _ = train_batch

# 创建Explainer实例
e = shap.GradientExplainer(model, train_images)
shap_values = e.shap_values(to_explain)

# index_names = []
# for x in indexes.numpy():
#     temp = []
#     for y in x:
#         temp.append(class_names[y])
#     temp = np.array(temp)
#     index_names.append(temp)
# index_names = np.array(index_names)
new_to_explain =np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
#可视化shap_value
save_path='result/mnist/'
shap.image_plot(shap_values, new_to_explain,show = False)
plt.savefig(save_path + 'epoch5_pool1_1.png')



