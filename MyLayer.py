import numpy as np

class MyLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, in_grad):
        raise NotImplementedError

    def update_params(self, lr):
        raise NotImplementedError


class ReLu_Layer(MyLayer):
    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, 0)

    def backward(self, in_grad):
        return np.where(self.input > 0, in_grad, 0)


class FC_Layer(MyLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

        self.weight = np.random.normal(loc=0, scale=0.01, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))
        # self.weight = np.random.rand(input_size, output_size) / input_size
        # self.bias = np.random.rand(1, output_size) / input_size

    def forward(self, input):
        self.input = input
        output = (self.input @ self.weight) + self.bias
        return output

    def backward(self, in_grad):
        # in_grad: m * output
        # d_weight = (input * m) * (m * output) = input * output
        self.grad_weight = (self.input.T @ in_grad)
        # d_bias = 1 * output
        self.grad_bias = np.sum(in_grad, axis=0)
        # d_input =  (m * output) * (output * input)= m * input
        return in_grad @ self.weight.T

    def update_params(self, lr):
        self.weight -= self.grad_weight * lr
        self.bias -= self.grad_bias * lr
        # print("grad", self.weight, self.grad_weight, self.bias, self.grad_bias)


