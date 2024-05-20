import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.Noise import gasuss_noise

def train(model, device, train_loader, optimizer, criterion, epoch, noiseGrad = 0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        #对梯度加噪
        if noiseGrad == 1:
            for param in model.parameters():
                gradients = param.grad
                gradients = gasuss_noise(gradients, var=gradients.detach().abs().mean().to('cpu') / 2)
                
        optimizer.step()
        print(
            f"Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)}"
            f" ({100.0 * (batch_idx+1) / len(train_loader):.0f}%)]"
            f"\tLoss: {loss.item():.6f}"
        )
    return loss.item()



def test(model, device, test_loader, criterion):
    model.eval()  # 设置模型为评估模式

    test_loss = 0.0
    correct = 0
    total = 0

    # 遍历测试数据集
    with torch.no_grad():  # 禁用梯度计算以节省内存和计算时间
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {100.0 * correct / total}%")


def preprocess_Data(path):
    batch_size = 128
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224])
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

