import numpy as np
import torch
from time import time
import os
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from util.Noise import gasuss_noise
from util.Model_Func import train,test
_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_class):
        super(VGG, self).__init__()
        self.features = self._make_layers(_cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class)

    # pylint: disable=W0221,E1101
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class noiseVGG(nn.Module):
    def __init__(self, vgg_name, num_class, loc):
        super(VGG, self).__init__()
        self.beFeatures, self.features = self._make_layers(_cfg[vgg_name], loc)
        self.classifier = nn.Linear(512, num_class)
    def forward(self, x):
        x = self.beFeatures(x)
        x = gasuss_noise(x, var=x.detach().abs().mean().to('cpu')/2)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, loc):
        beLayers = []
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                if i<loc:
                    beLayers+= [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i<loc:
                    beLayers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*beLayers), nn.Sequential(*layers)


def Preprocess_Data():
    batch_size = 640
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    # 数据预处理
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data/cifar10_data",
            train=True,
            download=False,
            transform=transform_train
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data/cifar10_data", train=False, transform=transform_test
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader,test_loader

#调整学习率
def lr_lambda(epoch):
    v = 0.0
    if epoch<40:
        v = 0.1
    elif epoch<80:
        v = 0.01
    else:
        v = 0.001
    return v


if __name__ =="__main__":

    #读取数据
    train_loader, test_loader = Preprocess_Data()
    # 训练参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    #无噪声模型
    model = VGG("VGG16",10).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=1,momentum=0.9,weight_decay=0.0005)
    # 学习率
    scheduler = LambdaLR(optimizer, lr_lambda)

    #训练模型
    losses = []
    times = []
    path = 'model/cifar10/no/'
    if not os.path.exists(path):
        os.makedirs(path)
    for epoch in range(1, num_epochs + 1):
        start= time()
        loss = train(model, device, train_loader, optimizer, criterion, epoch, noiseGrad=1)
        cost_time = time() - start
        losses.append(loss)
        times.append(cost_time)
        scheduler.step()
        # 保存模型
        if epoch % 20 == 0:
            torch.save(model.state_dict(), path + f"epoch{epoch}.pth")
            test(model, device, test_loader, criterion)
        if epoch % 100 == 0:
            if not os.path.exists(path + 'logs'):
                os.makedirs(path + 'logs')
                # 保存损失
            with open(os.path.join(path + 'logs', 'epoch' + str(epoch) + '_Losses.txt'), 'w') as f:
                for loss in losses:
                    f.write(f"{loss}\n")
                # 保存训练时间
            with open(os.path.join(path + 'logs', 'epoch' + str(epoch) + '_Times.txt'), 'w') as f:
                for Time in times:
                    f.write(f"{Time}\n")



