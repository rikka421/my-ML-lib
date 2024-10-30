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

    def get_params_grad(self):
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

        self.weight = np.random.normal(loc=0, scale=0.01, size=(output_size, input_size))
        self.bias = np.zeros((1, output_size))
        # self.weight = np.zeros((input_size, output_size))
        # self.weight = np.random.rand(input_size, output_size) / input_size
        # self.bias = np.ones((1, output_size))

    def forward(self, input):
        self.input = input
        output = (self.input @ self.weight.T) + self.bias
        return output

    def backward(self, in_grad):
        # in_grad: m * output
        # d_weight = (output * m) * (m * input) = output * input
        self.grad_weight = (in_grad.T @ self.input)
        # d_bias = 1 * output
        self.grad_bias = np.sum(in_grad, axis=0)
        # d_input =  (m * output) * (output * input)= m * input
        return in_grad @ self.weight

    def get_params(self):
        return [self.weight, self.bias]

    def get_grads(self):
        return [self.grad_weight, self.grad_bias]