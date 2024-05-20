import shap
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from STL import STLPreprocess_Data

# #shap值
# npPath = "result/stl/Shap_value/airplane13_car24/"
# shap_values = np.load(npPath + 'no_f0.npy')
# #可视化shap
# train_loader, test_loader = STLPreprocess_Data()
# indices = torch.tensor([13,24])
# test_batch = next(iter(test_loader))
# test_images, labels = test_batch
# to_explain = test_images[indices]
# new_to_explain = np.swapaxes(np.swapaxes(to_explain.numpy(), 1, -1), 1, 2)
# print(shap_values.shape)
# print(new_to_explain.shape)
#类名
x = [2.1258e-07, 2.8988e-08, 9.9891e-01, 6.8743e-06, 3.0306e-10, 1.5057e-07,
         1.5417e-09, 2.0712e-08, 2.7332e-09, 1.0846e-03]
res = 0
for i in x:
    res+=i

print(res)