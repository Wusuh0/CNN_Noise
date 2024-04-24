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
with open('data/stl10_binary/class_names.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]
class_names = np.array(class_names)

# 加载模型
model = Net()
model.load_state_dict(torch.load(path + "epoch5_pool1_mean.pth"))
print(model)
model.eval()

test_batch = next(iter(test_loader))
test_images, labels = test_batch
to_explain = test_images[:3]
train_batch = next(iter(train_loader))
train_images, _ = train_batch

# 创建Explainer实例
e = shap.GradientExplainer((model, model.conv2), train_images)
shap_values,indexes = e.shap_values(to_explain,ranked_outputs=2)
print(indexes)

index_names = indexes.numpy()

new_to_explain =np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
#可视化shap_value
save_path='result/stl/'
shap.image_plot(shap_values, new_to_explain, index_names,show =True)
# plt.savefig(save_path + 'epoch10_f2.png')



