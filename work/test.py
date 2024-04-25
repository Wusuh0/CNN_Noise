import torch
import numpy as np
from STL import STLPreprocess_Data,Net,noiseNet
from util.Model_Func import test
# 加载模型
train_loader, test_loader = STLPreprocess_Data()
model = Net()
model.load_state_dict(torch.load( "model/stl/epoch10.pth"))

test(model, test_loader)