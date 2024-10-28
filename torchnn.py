import torch
import torch.optim as optim
import numpy as np

from NeuralNetworks import NeuralNetwork
from NeuralNetworks import Layer

# 定义一个简单的神经网络
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义层：一个两层的全连接网络
        self.fc1 = torch.nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(128, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入（假设输入是28x28图像）
        x = torch.relu(self.fc1(x))  # 激活函数 ReLU
        x = self.fc2(x)  # 输出层
        return x

class MyNN(NeuralNetwork):
    def __init__(self):
        # 定义层：一个两层的全连接网络
        self.fc1 = Layer(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = Layer(128, 10, activation_function=lambda x:x)  # 隐藏层到输出层
        super(MyNN, self).__init__(28*28, 10, layers=[self.fc1, self.fc2])


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载训练和测试数据集
# m * (torch.Size([1, 28, 28]), int), 每个样本为(X, y)元组, 其中X是图片, y是标签
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def my_CrossEntropyLoss(logits, targets):
    sum_ans = 0
    for logit, target in zip(logits, targets):
        # 对于[10], int. 先计算sum(exp(p_i)), 再计算 exp(tar) / sum
        exp_sum = np.sum(np.exp(logit))
        target_val = np.exp(logit[target])
        soft_max_val = target_val / exp_sum
        sum_ans += np.log(soft_max_val)
    sum_ans /= len(targets)
    return -sum_ans

def my_train(model, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 前向传播
            X = np.array([image.view(-1).numpy() for image in images])
            Y = []
            for label in labels:
                array_label = [0.0] * 10
                array_label[label] = 1.0
                Y.append(array_label)
            Y = np.array(Y)
            # print(X.shape, Y.shape)
            model.train(X, Y, epoch=1, lr=0.001)
            tar_labels = np.array(labels)
            loss = my_CrossEntropyLoss(model.forward(X), tar_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

model = MyNN()
my_train(model, train_loader)


def evaluate(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 前向传播
            X = np.array([image.view(-1).numpy() for image in images])
            Y = model(X)
            _, predicted = np.max(Y, 1)
            total += labels.size(0)
            print(predicted, labels)
            correct += (predicted == labels).sum()

    print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清除上一步的梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 创建网络实例
model = SimpleNN()

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失（用于分类任务）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 开始训练
# train(model, train_loader, criterion, optimizer)


def evaluate(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')

# 评估模型
# evaluate(model, test_loader)
