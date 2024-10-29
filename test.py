import numpy as np
import matplotlib.pyplot as plt

from Layer import *
from Criterion import *

if __name__ == '__main__':
    np.random.seed(42)
    input_size = 1
    output_size = 1
    data_size = 50

    X = np.random.rand(data_size, input_size)
    b = np.random.rand(1, output_size)
    W = np.random.rand(input_size, output_size)
    Y = W * X + b

    fc = FC_Layer(input_size, output_size)
    criterion = SquareLoss()

    for i in range(100):
        pre_Y = fc.forward(X)
        loss = criterion.forward(pre_Y, Y)
        in_grad = criterion.backward(loss)
        fc.backward(in_grad)
        fc.update_params(lr=1e-2)

        plt.scatter(X, Y)
        plt.scatter(X, pre_Y)
        # plt.title("散点图")
        # plt.xlabel("X 轴")
        # plt.ylabel("Y 轴")
        plt.show()
