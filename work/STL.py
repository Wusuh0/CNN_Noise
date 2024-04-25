import numpy as np
import torch
import time
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from util.Model_Func import train,test
from util.Noise import gasuss_noise


class Net(nn.Module):
    def __init__(self,init_weight = True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=10)
        )
        self.softmax = nn.Softmax(dim=1)

        # 参数初始化
        if init_weight:  # 如果进行参数初始化
            for m in self.modules():  # 对于模型的每一层
                if isinstance(m, nn.Conv2d):  # 如果是卷积层
                    # 使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):  # 如果是线性层
                    # 正态初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    nn.init.constant_(m.bias, 0)



    def forward(self,x):
        x = self.conv1(x)
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.softmax(self.classifier(x))
        return x

class noiseNet(Net):
    def forward(self,x):
        x = self.conv1(x)
        x = gasuss_noise(x, var=x.detach().abs().mean())
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.softmax(self.classifier(x))
        return x

def STLPreprocess_Data():
    batch_size = 128
    train_loader = DataLoader(
        datasets.STL10(root="data", split="train", download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        datasets.STL10(root="data", split="test", download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
    )
    return train_loader,test_loader
if __name__ =="__main__":

    #读取数据
    train_loader, test_loader = STLPreprocess_Data()
    #初始化模型
    path = 'model/stl/'
    model = Net()
    model.load_state_dict(torch.load(path + "epoch20.pth"))
    model.to(torch.device('cuda:1'))
    print(model)
    # 定义损失函数，优化器和训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    num_epochs = 20
    torch.set_num_threads(8)
    # 训练模型
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train(model, train_loader, optimizer, criterion, epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch {epoch}训练时间: {elapsed_time}秒")
        test(model,  test_loader)
        # 保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), path + 'epoch'+str(20+epoch)+'.pth')

