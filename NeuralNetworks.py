import numpy as np


class NeuralNetwork:
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
