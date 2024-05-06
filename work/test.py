import torch
import numpy as np
import os
from torch import nn
from STL import Net,noiseNet,STLPreprocess_Data
from util.Model_Func import test,preprocess_Data

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载模型
train_loader, test_loader = STLPreprocess_Data()
model =Net().to(device)
#map_location=lambda storage, loc: storage.cuda(0)
model.load_state_dict(torch.load( "model/stl/epoch400.pth",map_location=lambda storage, loc: storage.cuda(0)))

test(model, device, test_loader, criterion)

