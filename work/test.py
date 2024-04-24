from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor



test_batch = next(iter(test_loader))
