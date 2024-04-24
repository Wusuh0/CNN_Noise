import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader

def train(model, trainloader, optimizer, criterion, epoch):
    model.train()
    # 记录训练时间
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播 + 反向传播 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 打印损失
        if i % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{i * len(data)}/{len(trainloader.dataset)}"
                f" ({10.0 * i / len(trainloader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


def test(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():

        for images, labels in testloader:
            outputs = model(images)
            test_loss += F.nll_loss(outputs.log(), labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            f"\nTest set: Average loss: {test_loss:.4f},"
            f" Accuracy: {correct}/{len(testloader.dataset)}"
            f" ({100.0 * correct / len(testloader.dataset):.0f}%)\n"
        )


def preprocess_Data(path):
    batch_size = 64
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载数据集
    Train_data = datasets.ImageFolder(root= path + 'train',
                                       transform=transform,
                                                  )
    # 创建数据加载器
    Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)

    Test_data = datasets.ImageFolder(root= path + 'test',
                                  transform=transform)
    Test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)

    return Train_loader, Test_loader

