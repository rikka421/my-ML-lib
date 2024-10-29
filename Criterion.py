import numpy as np

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
