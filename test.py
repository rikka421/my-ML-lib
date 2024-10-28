from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    # transforms.Normalize((0.5,), (0.5,))  # 归一化
    transforms.Lambda(lambda x: x.view(-1)),  # 张量转换为一维
])
target_transform = transforms.Compose([
    transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)),  # onehot
])

# 加载训练和测试数据集
# m * (torch.Size([1, 28, 28]), int), 每个样本为(X, y)元组, 其中X是图片, y是标签
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
    target_transform=target_transform
    )
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform,
    target_transform=target_transform
    )

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)