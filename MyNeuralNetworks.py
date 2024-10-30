import numpy as np


class MyNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = None
        self.update_layer_list = None

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, in_grad):
        for layer in self.layers[::-1]:
            in_grad = layer.backward(in_grad)

    def step(self, lr=1e-3):
        for layer in self.update_layer_list:
            layer.update_params(lr)

    def get_params(self):
        params = []
        for layer in self.update_layer_list:
            param = layer.get_params()
            params += param
        return params

    def get_grads(self):
        grads = []
        for layer in self.update_layer_list:
            grad = layer.get_grads()
            grads += grad
        return grads