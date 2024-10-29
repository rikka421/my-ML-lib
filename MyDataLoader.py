import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MyDataLoader:
    def __init__(self, input_size, output_size, batch_num, batch_size):
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.cur_batch = 0

        self.inputs = None
        self.labels = None

    def get_data_set(self):
        return self.inputs, self.labels

    def __iter__(self):
        # 返回可迭代对象本身
        return self

    def __next__(self):
        # 如果当前计数器超过上限，抛出 StopIteration 异常
        if self.cur_batch >= self.batch_num:
            self.cur_batch = 0
            raise StopIteration
        else:
            # 返回当前值，并将计数器加一
            value = (self.inputs[self.cur_batch * self.batch_size : (self.cur_batch + 1) * self.batch_size],
                     self.labels[self.cur_batch * self.batch_size : (self.cur_batch + 1) * self.batch_size])
            self.cur_batch += 1
            return value

class LinearData(MyDataLoader):
    def __init__(self, input_size, output_size, batch_num, batch_size):
        super(LinearData, self).__init__(input_size, output_size, batch_num, batch_size)

        data_size = batch_num * batch_size
        X = np.random.rand(data_size, input_size)
        W = np.random.rand(input_size, output_size)
        b = np.random.rand(1, output_size)
        Y = X @ W - b

        self.inputs = X
        self.labels = Y

class ReLuData(MyDataLoader):
    def __init__(self, input_size, output_size, batch_num, batch_size):
        super(ReLuData, self).__init__(input_size, output_size, batch_num, batch_size)

        data_size = batch_num * batch_size
        X = np.random.rand(data_size, input_size) * 2
        W = np.ones((input_size, output_size)) / input_size
        b = np.ones((1, output_size))
        Y = X @ W - b
        Y = np.where(Y > 0, Y, 0)

        self.inputs = X
        self.labels = Y


class SquareData(MyDataLoader):
    def __init__(self, input_size, output_size, batch_num, batch_size):
        super(SquareData, self).__init__(input_size, output_size, batch_num, batch_size)

        assert output_size == 1

        data_size = batch_num * batch_size
        X = np.random.rand(data_size, input_size)
        Q = np.random.rand(input_size, input_size)
        Q = (Q + Q.T) / 2
        b = np.random.rand(1, output_size)
        # 计算每个样本的结果
        # 先计算 XQ
        XQ = X @ Q  # 结果形状为 (m, d)

        # 然后计算 xQx + b
        # 使用 np.einsum 进行高效计算
        Y = np.einsum('ij,ik->i', X, XQ) - b  # 结果形状为 (m,)
        Y = Y.reshape(data_size, 1)

        self.inputs = X
        self.labels = Y

class MNISTData(MyDataLoader):
    def __init__(self, train=True):
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
        if train:
            dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform,
                target_transform=target_transform
                )
        else:
            dataset = datasets.MNIST(
                root='./data',
                train=False,
                download=True,
                transform=transform,
                target_transform=target_transform
                )

        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
        batch_num = len(loader)

        inputs_list = []
        labels_list = []

        for inputs, labels in loader:
            # 将输入和标签添加到列表中
            inputs_list.append(inputs.numpy())  # 转换为 NumPy 数组
            labels_list.append(labels.numpy())  # 转换为 NumPy 数组

        super(MNISTData, self).__init__(28 * 28, 10, batch_num, 64)

        # 将列表转换为 NumPy 数组
        self.inputs = np.concatenate(inputs_list, axis=0)  # shape: (num_samples, 1, 28, 28)
        self.labels = np.concatenate(labels_list, axis=0)



