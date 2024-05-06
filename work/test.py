import torch
import numpy as np
import os
from torch import nn
from STL import Net,noiseNet,STLPreprocess_Data
from util.Model_Func import test,preprocess_Data

with open('data/stl10_binary/class_names.txt', 'r') as file:
    class_list = [line.strip() for line in file.readlines()]
class_names = np.array(class_list)
train_loader, test_loader = STLPreprocess_Data()
test_batch = next(iter(test_loader))
images ,labels = test_batch
indices = torch.tensor([13,18,19,24,29,42,49])
print(np.vectorize(lambda x: class_names[x])(labels[indices]))

# model = Net()
# model.load_state_dict(torch.load( "model/stl/epoch100.pth",map_location=lambda storage, loc: storage.cpu()))
# model_noise = noiseNet()
# model_noise.load_state_dict(torch.load( "model/stl/epoch100conv1_0.5mean.pth",map_location=lambda storage, loc: storage.cpu()))
# criterion = nn.CrossEntropyLoss()
# device = torch.device("cpu")
# test(model,device,test_loader,criterion)
# test(model_noise,device,test_loader,criterion)
