import numpy as np
from torch.utils.data import Dataset


class DataLoader:
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

class LinearData(DataLoader):
    def __init__(self, input_size, output_size, batch_num, batch_size):
        super(LinearData, self).__init__(input_size, output_size, batch_num, batch_size)

        data_size = batch_num * batch_size
        X = np.random.rand(data_size, input_size)
        W = np.random.rand(input_size, output_size)
        b = np.random.rand(1, output_size)
        Y = W * X - b

        self.inputs = X
        self.labels = Y

class ReLuData(DataLoader):
    def __init__(self, input_size, output_size, batch_num, batch_size):
        super(ReLuData, self).__init__(input_size, output_size, batch_num, batch_size)

        data_size = batch_num * batch_size
        X = np.random.rand(data_size, input_size) * 2
        W = np.ones((input_size, output_size)) / input_size
        b = np.ones((1, output_size))
        Y = W * X - b
        Y = np.where(Y > 0, Y, 0)

        self.inputs = X
        self.labels = Y


class SquareData(DataLoader):
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







