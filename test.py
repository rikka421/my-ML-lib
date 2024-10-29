import numpy as np
import matplotlib.pyplot as plt

from DataLoader import *
from Layer import *
from Criterion import *

if __name__ == '__main__':
    np.random.seed(42)

    input_size = 1
    output_size = 1
    batch_num = 100
    batch_size = 64
    data_size = batch_size * batch_num

    data_loader = SquareData(input_size, output_size, batch_num, batch_size)
    X, Y = data_loader.get_data_set()

    fc = FC_Layer(input_size, output_size)
    criterion = SquareLoss()

    epochs = 5
    for epoch_i in range(epochs):
        for batch_i, (input_data, label) in enumerate(data_loader):
            # print(batch_i, input_data.shape, label.shape)
            pre_label = fc.forward(input_data)
            loss = criterion.forward(pre_label, label)
            in_grad = criterion.backward(loss)
            fc.backward(in_grad)
            fc.update_params(lr=1e-3)
        print(epoch_i)
        pre_Y = fc.forward(X)
        plt.scatter(X, Y)
        plt.scatter(X, pre_Y)
        plt.show()
