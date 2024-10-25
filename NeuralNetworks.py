import numpy as np

class Layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # weights: input * output
        # self.weights = np.ones((input_size, output_size))
        # self.bias = np.ones((1, output_size))
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def activation(self, x):
        # 阶跃函数作为激活函数
        return np.where(x > 0, x, 0)

    def activation_derivative(self, linear_out, pre_d_value):
        # x: m * input
        # 阶跃函数作为激活函数的导数
        # return m * output * input
        return np.where(linear_out > 0, pre_d_value, 0)

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.bias
        self.outputs = self.activation(self.linear_output)
        return self.outputs

    def backward(self, pre_d_value):
        # pre_d_value: m * output
        # output2linear: m * output
        output2linear = self.activation_derivative(self.linear_output, pre_d_value)
        # d_value: (m * output * output) @ (m * output) = m * output
        # print(output2linear.shape)
        # print(output2linear.shape)
        # print(self.inputs.shape)
        # d_weight: output @ input = m * output
        d_weight = (self.inputs.T @ output2linear)
        # print(d_weight.shape)
        d_bias = np.sum(output2linear, axis=0)
        self.weights = self.weights - d_weight * 0.01
        self.bias = self.bias - d_bias * 0.01

        return d_weight, d_bias


if __name__ == '__main__':
    np.random.seed(42)
    layer1 = Layer(3, 3)
    # print(layer1.weights)
    # print(layer1.bias)

    inputs = np.random.rand(10, 3)
    y = inputs * 2
    for i in range(10000):
        l1 = layer1.forward(inputs)
        layer1.backward(l1 - y)
        print(np.sum(np.square(l1 - y) / 2))


