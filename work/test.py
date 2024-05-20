import torch
import numpy as np
import os
from torch import nn
from STL import Net,noiseNet,STLPreprocess_Data
from util.Model_Func import test,preprocess_Data

# 读取数据
train_loader, test_loader = STLPreprocess_Data()
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
model = Net()
model.load_state_dict(torch.load( "model/stl/grad_0.5mean/epoch40grad_0.5mean.pth",map_location=lambda storage, loc: storage.cpu()))
test(model, device,test_loader,criterion)
