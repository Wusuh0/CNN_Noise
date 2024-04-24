import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from util.Noise import gasuss_noise


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(500, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(self.dropout(x))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class noiseNet(Net):
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = gasuss_noise(x, var=x.detach().abs().mean())
        x = self.pool2(self.conv2(x))
        x = self.flatten(self.dropout(x))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )
def MNISTPreprocess_Data():
    batch_size = 128
    # 数据预处理
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist_data", train=False, transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader,test_loader

if __name__ =="__main__":
    # 训练参数

    num_epochs = 5
    device = torch.device("cpu")
    train_loader, test_loader = MNISTPreprocess_Data()

    # 无噪声模型
    # model = Net().to(device)
    #
    # for epoch in range(1, num_epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    #
    # torch.save(model.state_dict(), 'model/mnist/epoch5.pth')

    # 加噪模型(控制噪声加入的位置和大小)
    noiseModel = noiseNet().to(device)
    optimizer = optim.SGD(noiseModel.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, num_epochs + 1):
        train(noiseModel, device, train_loader, optimizer, epoch)
        test(noiseModel, device, test_loader)

    torch.save(noiseModel.state_dict(), 'model/mnist/epoch5_pool1_mean.pth')