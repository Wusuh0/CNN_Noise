import shap
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from Cifar10 import Net,noiseNet
from util.Model_Func import preprocess_Data
from Cifar10 import Cifar10Preprocess_Data
# 加载数据集
train_loader, test_loader = Cifar10Preprocess_Data()
path = "model/cifar10/"

#加载类别
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 加载模型
model = noiseNet()
model.load_state_dict(torch.load(path + "epoch20_conv1_mean.pth"))
print(model)
model.eval()

test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[5:8]
train_batch = next(iter(train_loader))
train_images, _ = train_batch
print(np.vectorize(lambda x: class_names[x])(labels[5:8]))
# 创建Explainer实例
e = shap.GradientExplainer((model,model.pool1), train_images)
shap_values , indexes = e.shap_values(to_explain,ranked_outputs = 2)
print(np.array(shap_values).shape)
index_names = np.vectorize(lambda x: class_names[x])(indexes)

new_to_explain =np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

#可视化shap_value
shap.image_plot(shap_values, new_to_explain,index_names,show = False)
# show = False
save_path='result/cifar10/conv1_mean/'
plt.savefig(save_path + 'pool1.png')



