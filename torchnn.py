import numpy as np
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

import torch.optim as optim


# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=5, my_model=True):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 前向传播
            if my_model:
                images = images.numpy()
                outputs = model.forward(images)
                outputs = torch.from_numpy(outputs)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)

            if my_model:
                # Loss = -1/N (y - log sum e^hat_y)
                # d log sum e^hat_y / d hat_y = e^hat_y / log sum e^hat_y
                labels = labels.numpy()
                pre_d_value = - np.sum(labels)
                pre_d_value += np.exp(outputs) / np.log(np.sum(np.exp(outputs)))
                pre_d_value /= len(labels)
                model.backward(pre_d_value, lr=1e-3)
            else:
                # 反向传播和优化
                optimizer.zero_grad()  # 清除上一步的梯度
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate(model, test_loader, my_model=True):
    if not my_model:
        model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if my_model:
                images = images.numpy()
                outputs = model.forward(images)
                outputs = torch.from_numpy(outputs)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')


inner_size = 128

from NeuralNetworks import NeuralNetwork
from NeuralNetworks import Layer

# 定义自己的神经网络
class MyNN(NeuralNetwork):
    def __init__(self, input_size, output_size):
        # 定义层：一个两层的全连接网络
        self.fc1 = Layer(input_size, inner_size)  # 输入层到隐藏层
        self.fc2 = Layer(inner_size, output_size, activation_function=lambda x:x)  # 隐藏层到输出层
        super(MyNN, self).__init__(input_size, output_size, [self.fc1, self.fc2])


# 创建网络实例
model = MyNN(28*28, 10)
optimizer = None  # 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失（用于分类任务）

# 开始训练
train(model, train_loader, criterion, optimizer, my_model=True)