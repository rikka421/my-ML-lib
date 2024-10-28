import numpy as np

class Layer():
    def __init__(self, input_size, output_size, activation_function=None):
        self.input_size = input_size
        self.output_size = output_size

        # weights: input * output
        # self.weights = np.random.rand(input_size, output_size) / input_size
        # self.bias = np.random.rand(1, output_size)
        self.weights = np.ones((input_size, output_size)) / input_size
        self.bias = np.ones((1, output_size)) / input_size

        self.activation_function = activation_function

        self.inputs = None
        self.linear_output = None
        self.outputs = None

    def activation(self, x):
        # 阶跃函数作为激活函数
        if self.activation_function is None:
            # return 1 / (1 + np.exp(-x))
            return np.where(x > 0, x, 0)
        return x


    def activation_derivative(self, linear_out, pre_d_value):
        # x: m * input
        # 阶跃函数作为激活函数的导数
        # return m * output * input
        if self.activation_function is None:
            # return pre_d_value * (1 - pre_d_value)
            return np.where(linear_out > 0, pre_d_value, 0)
        return pre_d_value

    def forward(self, inputs):
        self.inputs = inputs
        # print("self.inputs.shape", self.inputs.shape)
        # print("self.weights", self.weights.shape)
        # print("self.bias", self.bias.shape)
        self.linear_output = (self.inputs @ self.weights) + self.bias
        # print("self.linear_output.shape", self.linear_output.shape)
        self.outputs = self.activation(self.linear_output)
        # print("self.outputs.shape", self.outputs.shape)
        return self.outputs

    def backward(self, pre_d_value, lr):
        # output2linear = m * output
        output2linear = self.activation_derivative(self.linear_output, pre_d_value)
        # d_weight = (input * m) * (m * output) = input * output
        d_weight = (self.inputs.T @ output2linear) / len(self.inputs)
        # d_bias = 1 * output
        d_bias = np.sum(output2linear, axis=0) / len(self.inputs)
        # d_input =  (m * output) * (output * input)= m * input
        d_input = output2linear @ self.weights.T
        # print("d", output2linear, d_weight, d_bias, d_input)

        # print(output2linear.shape, d_weight.shape, d_bias.shape)
        # print(d_input.shape)


        self.weights = self.weights - d_weight * lr
        self.bias = self.bias - d_bias * lr
        # print("mean", np.sum(d_weight) / d_weight.size,  np.sum(d_bias) / d_bias.size)

        return d_input


class NeuralNetwork:
    def __init__(self, input_size, output_size, layers):
        self.layers = layers

    def forward(self, X):
        res = X
        for layer in self.layers:
            res = layer.forward(res)
            # print("res", res)
            # print("weight, bias", layer.weights, layer.bias)
        # print("res", res)
        return res

    def backward(self, Y, pre_Y, lr):
        # print("pre_Y, Y", pre_Y, Y)
        pre_d_values = (pre_Y - Y) / Y.shape[1]
        for layer in self.layers[::-1]:
            # print("pre_d", pre_d_values)
            # pre_d_values /= np.sum(pre_d_values)
            pre_d_values = layer.backward(pre_d_values, lr)

    def train(self, X, Y, epoch=10000, lr=1e-3):

        for i in range(epoch):
            pre_Y = self.forward(X)
            self.backward(Y, pre_Y, lr)
            loss = np.sum(np.square(Y - pre_Y)) / Y.size
            if i % 200 == 0:
                print(loss)
            if loss < 0.001:
                print("训练完成！ 第%d次迭代" % (i))
                break

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            # print(X)
        return self.forward(X)


def test():
    pass

if __name__ == '__main__':
    np.random.seed(42)

    m = 1000
    d = 100
    inner = 128

    X = np.random.rand(m, d)
    # X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = 2 * X

    fc1 = Layer(d, inner)
    inner_num = 1
    fcs = [Layer(inner, inner)] * inner_num
    fc2 = Layer(inner, d, activation_function=lambda x: x)
    fc_one = Layer(d, d, activation_function=lambda x: x)

    # print([fc1] + [fc2])

    # model = NeuralNetwork
    model = NeuralNetwork(d, d, [fc1]+fcs+[fc2])

    model.train(X, Y)