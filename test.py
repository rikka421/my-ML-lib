import numpy as np
import matplotlib.pyplot as plt
from sspicon import SEC_E_NO_PA_DATA

from DataLoader import *
from DataLoader import LinearData
from Layer import *
from Criterion import *
from NeuralNetworks import *

class SimpelModel(NeuralNetwork):
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


if __name__ == '__main__':
    np.random.seed(42)

    input_size = 1
    inner_sizes = [4]
    output_size = 1
    batch_num = 1000
    batch_size = 64
    data_size = batch_size * batch_num

    data_loader = SquareData(input_size, output_size, batch_num, batch_size)
    X, Y = data_loader.get_data_set()

    model = SimpelModel(input_size, inner_sizes, output_size)
    print(model.layers, model.update_layer_list)

    criterion = SquareLoss()

    epochs = 5
    for epoch_i in range(epochs):
        for batch_i, (input_data, label) in enumerate(data_loader):
            # print(batch_i, input_data.shape, label.shape)
            pre_label = model.forward(input_data)
            loss = criterion.forward(pre_label, label)
            in_grad = criterion.backward(loss)
            # print(np.max(in_grad))
            model.backward(in_grad)
            # print(np.max(in_grad))
            model.step()
        print(epoch_i)

        pre_Y = model.forward(X)

        plt.scatter(X, Y)
        plt.scatter(X, pre_Y)
        plt.show()
