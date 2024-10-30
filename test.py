import numpy as np
import matplotlib.pyplot as plt
import time

from MyDataLoader import *
from MyCriterion import *
from MyOptimizer import *
from MyLayer import *
from MyNeuralNetworks import *


class SimpelModel(MyNeuralNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpelModel, self).__init__(input_size, hidden_sizes, output_size)

        self.layers = []
        self.update_layer_list = []
        if hidden_sizes:
            fc_in = FC_Layer(input_size, hidden_sizes[0])
            rel = ReLu_Layer(hidden_sizes[0], hidden_sizes[0])
            self.layers.append(fc_in)
            self.update_layer_list.append(fc_in)
            self.layers.append(rel)
            for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                fc = FC_Layer(in_size, out_size)
                rel = ReLu_Layer(out_size, out_size)
                self.layers.append(fc)
                self.update_layer_list.append(fc)
                self.layers.append(rel)
            fc_out = FC_Layer(hidden_sizes[-1], output_size)
            self.layers.append(fc_out)
            self.update_layer_list.append(fc_out)
        else:
            fc_in = FC_Layer(input_size, output_size)
            self.layers = [fc_in]
            self.update_layer_list = [fc_in]

def test_MNIST():
    print("begin test CrossEntropyLoss and MNIST DataSet")
    np.random.seed(42)

    input_size = 28 * 28
    inner_sizes = [128]
    output_size = 10

    model = SimpelModel(input_size, inner_sizes, output_size)

    data_loader = MNISTData(train=True)
    test_data_loader = MNISTData(train=False)
    X, Y = test_data_loader.get_data_set()


    criterion = CrossEntropyLoss()

    epochs = 10
    for epoch_i in range(epochs):
        for batch_i, (input_data, label) in enumerate(data_loader):
            # print(batch_i)
            # print(np.max(input_data), np.min(input_data))
            pre_label = model.forward(input_data)
            # print(np.exp(pre_label[0]) / np.sum(np.exp(pre_label[0])), label[0])
            loss = criterion.forward(pre_label, label)
            in_grad = criterion.backward()
            # print(np.max(in_grad))
            model.backward(in_grad)
            # print(np.max(in_grad))
            model.step()
        print(epoch_i, loss)

        pre_Y = model.forward(X)
        ans = np.argmax(pre_Y, axis=1)
        labels = np.argmax(Y, axis=1)
        print(np.sum(ans == labels) / labels.size)
    print("朝乾夕惕, 功不唐捐")
    print("Congratulation!")

def test_CrossEntropyLoss():
    print("begin test CrossEntropyLoss and CrossEntropy DataSet")
    np.random.seed(42)

    input_size = 2
    inner_sizes = [128]
    output_size = 2
    batch_num = 1000
    batch_size = 64
    data_size = batch_size * batch_num

    model = SimpelModel(input_size, inner_sizes, output_size)

    data_loader = CrossEntropyData(input_size, output_size, batch_num, batch_size)
    X, Y = data_loader.get_data_set()

    criterion = CrossEntropyLoss()

    epochs = 10
    for epoch_i in range(epochs):
        for batch_i, (input_data, label) in enumerate(data_loader):
            pre_label = model.forward(input_data)
            loss = criterion.forward(pre_label, label)
            in_grad = criterion.backward()
            model.backward(in_grad)
            model.step()
        print(epoch_i, loss)

        pre_Y = model.forward(X)
        ans = np.argmax(pre_Y, axis=1)
        labels = np.argmax(Y, axis=1)
        print(np.sum(ans == labels) / labels.size)

        pre_Y = model.forward(X)

        plt.scatter(X, Y)
        plt.scatter(X, pre_Y)
        plt.show()

def test_function(function, inner_sizes=None):
    print("begin test SquareLoss and Function DataSet")
    input_size = 1
    inner_sizes = [128] if inner_sizes is None else inner_sizes
    output_size = 1
    batch_num = 800
    batch_size = 64
    data_size = batch_size * batch_num

    model = SimpelModel(input_size, inner_sizes, output_size)
    params = model.get_params()
    optimizer = Adam(params)
    # optimizer = SimpleSGD(params)
    criterion = SquareLoss()

    data_loader = FunctionData(input_size, output_size, batch_num, batch_size, function)
    X, Y = data_loader.get_data_set()

    epochs = 50
    for epoch_i in range(epochs):
        for batch_i, (input_data, label) in enumerate(data_loader):
            pre_label = model.forward(input_data)
            loss = criterion.forward(pre_label, label)
            in_grad = criterion.backward()
            model.backward(in_grad)
            grads = model.get_grads()
            # model.step(1e-3)
            optimizer.step(grads)
        if epoch_i % 10 == 0:
            print(epoch_i, f"{loss:4f}")
        if loss <= 1e-5:
            print("loss <= 1e-5, break!")
            break

        pre_Y = model.forward(X)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(X[:1000, :], Y[:1000, :])
        plt.scatter(X[:1000, :], pre_Y[:1000, :])
        plt.show()

def opi_func(x):
    x = x * 3 - 1
    res1 = (np.sqrt((abs(1 - x)- x + 1) / 2) + 1 / 4) * np.exp(-(1 - x) ** 2)
    res2 = abs(1 - 2 ** 4 * (5 * x - 3) ** 4)
    res3 = abs(1 - 2 ** 9 * (5 * x - 3) ** 4)
    res4 = -528 * (5 * x - 3) ** 4 + 2

    return (res1 + (res2 + res3 + res4) / 40) / 2

def test_some_functions():
    func_lst = [
        # lambda x:x,
        # lambda x: (x * 2 - 1) ** 2,
        # lambda x: np.sin(x * 2 * np.pi) / 2 + 1 / 2,
        opi_func,
    ]

    for func in func_lst:
        for inner_sizes in [
            # [],
            # [4],
            # [16],
            # [64],
            [128],
            # [256],
            [512],
            # [1024],
            # [4, 4],
            # [16, 16],
            [64, 64],
            # [128, 128]
            [256, 256],
            # [512, 512]
        ]:
            print("中间层数量:", inner_sizes)
            test_function(func, inner_sizes=inner_sizes)
            time.sleep(1)

if __name__ == '__main__':
    # test_MNIST()
    # test_CrossEntropyLoss()
    test_some_functions()

