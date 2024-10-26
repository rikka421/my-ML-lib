import numpy as np

class Layer():
    def __init__(self, input_size, output_size, activation_function=None):
        self.input_size = input_size
        self.output_size = output_size

        # weights: input * output
        # self.weights = np.ones((input_size, output_size))
        # self.bias = -10 * np.ones((1, output_size))
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.ones((1, output_size))
        # print(self.weights, self.bias)

        self.activation_function = activation_function

    def activation(self, x):
        # 阶跃函数作为激活函数
        if self.activation_function is None:
            return np.where(x > 0, x, 0)
        return x


    def activation_derivative(self, linear_out, pre_d_value):
        # x: m * input
        # 阶跃函数作为激活函数的导数
        # return m * output * input
        if self.activation_function is None:
            return np.where(linear_out > 0, pre_d_value, 0)
        return pre_d_value

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.bias
        self.outputs = self.activation(self.linear_output)
        return self.outputs

    def backward(self, pre_d_value, lr):
        # output2linear = m * output
        output2linear = self.activation_derivative(self.linear_output, pre_d_value)
        # d_weight = input * output
        d_weight = (self.inputs.T @ output2linear)
        # d_bias = 1 * output
        d_bias = np.sum(output2linear, axis=0)
        # d_input =  (m * output) * (output * input)= m * input
        d_input = output2linear @ self.weights.T

        # print(output2linear.shape, d_weight.shape, d_bias.shape)
        # print(d_input.shape)

        # print(d_weight, d_bias)

        self.weights = self.weights - d_weight * lr
        self.bias = self.bias - d_bias * lr

        return d_input


class NeuralNetwork:
    def __init__(self, input_size, output_size, layers):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers

        self.X = None
        self.Y = None

    def forward(self, X):
        self.X = X
        for layer in self.layers:
            X = layer.forward(X)
        self.Y = X
        return X

    def backward(self, X, Y, lr):
        pre_d_values = self.Y - Y
        for layer in self.layers[::-1]:
            pre_d_values = layer.backward(pre_d_values, lr)

    def train(self, X, Y, epoch=1000, lr=0.01):
        for i in range(epoch):
            self.forward(X)
            self.backward(X, Y, lr)
            print(np.sum(np.square(Y - self.Y)))

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            # print(X)
        return self.forward(X)

def test1():
    np.random.seed(42)
    layer1 = Layer(3, 3)
    # print(layer1.weights)
    # print(layer1.bias)

    inputs = np.random.rand(10, 3)
    y = inputs * 2
    for i in range(10000):
        l1 = layer1.forward(inputs)
        layer1.backward(l1 - y, 0.01)
        print(np.sum(np.square(l1 - y)))

def test2():
    d = 2
    n = 100

    X = np.random.rand(100, d)
    Y = X * X

    layers = [
        # Layer(d, d),
        Layer(d, n),
        # Layer(n, n),
        Layer(n, d, activation_function=lambda x: x),
        # Layer(n, d, activation_function=lambda x: x)
    ]

    nn = NeuralNetwork(layers[0].input_size, layers[-1].output_size, layers)

    # print(nn.forward(X))

    nn.train(X, Y, 500, 1e-4)
    nn.predict(X)

def test():
    test2()

if __name__ == '__main__':
    np.random.seed(42)

    test()

