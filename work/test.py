import torch
import numpy as np
from Flower import Net,noiseNet
from util.Model_Func import test,preprocess_Data
# 加载模型
train_loader, test_loader = preprocess_Data("data/flower/")
model = Net()
model.load_state_dict(torch.load( "model/flower/epoch20.pth"))

test(model, test_loader)