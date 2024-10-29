import numpy as np

class NewDataLoader():
    def __init__(self):
        self.batch_nums = batch_num
        self.batch_size = batch_size
        self.input_data = np.random.rand(self.batch_size * self.batch_nums, input_size)
        # self.input_label = np.sum(self.input_data, axis=1).reshape((self.batch_size * self.batch_nums, 1)) @ np.ones(output_size).reshape((1, output_size))
        
        # self.input_label /= input_size
        self.input_label = 2 * self.input_data


    def get_data(self, i):
        return self.input_data[self.batch_size * i:self.batch_size * (i + 1)], self.input_label[self.batch_size  * i:self.batch_size * (i + 1)]
    
    def shuffle_data(self):
        pass
    


class Criterion():
    def __init__(self):
        self.input = None
        self.label = None

    def forward(self, inputs, labels):
        raise NotImplementedError
    
    def backward(self, in_grad):
        raise NotImplementedError

class SquareLoss(Criterion):
    def forward(self, input, label):
        # input: m*output  label: m * output
        self.input = input
        self.label = label
        # print(self.input.shape, self.label.shape)

        outputs = np.sum(np.square(input - label)) / (2 * label.size)
        # print(np.max(input), np.max(label), np.max(outputs))
        return outputs

    def backward(self, in_grad):
        # return m * output
        return (self.input - self.label) / (self.label.shape[1])


class Layer():
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
    


class ReLu_Layer(Layer):
    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, 0)
    
    def backward(self, in_grad):
        return np.where(self.input > 0, in_grad, 0)


class FC_Layer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        
        self.weight = np.random.rand(input_size, output_size) / input_size
        self.bias = np.random.rand(1, output_size) / input_size

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


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        if hidden_sizes:
            self.layers = []
            self.hiddens = []

            self.fc_in = FC_Layer(input_size, hidden_sizes[0])
            rel = ReLu_Layer(hidden_sizes[0], hidden_sizes[0])
            self.layers.append(self.fc_in)
            self.layers.append(rel)
            for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                fc = FC_Layer(in_size, out_size)
                rel = ReLu_Layer(out_size, out_size)
                self.hiddens.append(fc)
                self.layers.append(fc)
                self.layers.append(rel)
            self.fc_out = FC_Layer(hidden_sizes[-1], output_size)
            self.layers.append(self.fc_out)
        else:
            self.fc_in = FC_Layer(input_size, output_size)
            self.layers = [self.fc_in]
    
        self.update_layer_list = [self.fc_in] + self.hiddens + [self.fc_out]

    def forward(self, X):
        # print("X", X)
        for layer in self.layers:
            X = layer.forward(X)
            # print("X", X)
        return X

    def backward(self, in_grad):
        # print("pre_Y, Y", pre_Y, Y
        # print("grad, ", in_grad)
        for layer in self.layers[::-1]:
            in_grad = layer.backward(in_grad)
            # print("grad, ", in_grad)
    
    def step(self, lr):
        for i, layer in enumerate(self.update_layer_list):
            layer.update_params(lr)
            # print(i, "weight", layer.grad_weight)
            # print(i, "bias", layer.grad_bias)

    def train(self, train_data_loader, criterion, epochs=5, lr=2e-4):
        for idx_epoch in range(epochs):
            train_data_loader.shuffle_data()
            # 训练
            for id_1 in range(train_data_loader.batch_nums):
                train_data, train_labels = train_data_loader.get_data(id_1)
                # 前向传播
                output = self.forward(train_data)
                # print(output.shape, train_labels.shape)
                loss = criterion.forward(output, train_labels)
                # print(loss, np.max(output), np.max(train_labels))
                #exit(0)
                # 反向传播
                dloss = criterion.backward(loss)
                self.backward(dloss)
                # 参数更新
                self.step(lr)

                if id_1 % 10 == 0:
                    print("Train Epoch %d, iter %d, loss: %.6f" % (idx_epoch, id_1, loss))


if __name__ == '__main__':
    np.random.seed(42)

    batch_num = 1000
    batch_size = 64
    input_size = 10
    inner = 128
    output_size = 10

    model = NeuralNetwork(input_size, [inner], output_size)

    loader = NewDataLoader()

    criterion = SquareLoss()

    model.train(loader, criterion)