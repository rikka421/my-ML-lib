import numpy as np
import matplotlib.pyplot as plt

from MyDataLoader import *
from MyLayer import *
from MyCriterion import *
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
    np.random.seed(42)

    input_size = 28 * 28
    inner_sizes = [128]
    output_size = 10
    """batch_num = 1000
    batch_size = 64
    data_size = batch_size * batch_num"""

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


def test_Linear():
    np.random.seed(42)

    input_size = 1
    inner_sizes = [128]
    output_size = 1
    batch_num = 1000
    batch_size = 64
    data_size = batch_size * batch_num

    model = SimpelModel(input_size, inner_sizes, output_size)

    data_loader = SquareData(input_size, output_size, batch_num, batch_size)
    X, Y = data_loader.get_data_set()

    # criterion = CrossEntropyLoss()
    criterion = SquareLoss()

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

        plt.scatter(X, Y)
        plt.scatter(X, pre_Y)
        plt.show()



if __name__ == '__main__':
    print("begin test SquareLoss and Square DataSet")
    test_Linear()
    print("begin test CrossEntropyLoss and MNIST DataSet")
    test_MNIST()
    print("朝乾夕惕, 功不唐捐")
    print("Congratulation!")