import torch

import numpy as np

# 定义一个简单的神经网络
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义层：一个两层的全连接网络
        self.fc1 = torch.nn.Linear(d, inner_size)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(inner_size, d)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数 ReLU
        x = self.fc2(x)  # 输出层
        return x

import torch.optim as optim
# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=5000):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除上一步的梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')

m = 1000
d = 100
inner_size = 128

train_X = torch.from_numpy(np.array(np.random.rand(m, d), dtype=np.float32))
train_Y = train_X * 2
train_loader = [(train_X, train_Y)]
test_loader = train_loader

# 创建网络实例
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器# 损失函数和优化器
criterion = torch.nn.MSELoss()# torch.nn.CrossEntropyLoss()  # 交叉熵损失（用于分类任务）

# 开始训练
train(model, train_loader, criterion, optimizer)

# 评估模型
evaluate(model, test_loader)