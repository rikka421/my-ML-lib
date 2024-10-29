import numpy as np

class MyCriterion():
    def __init__(self):
        self.input = None
        self.label = None

    def forward(self, inputs, labels):
        raise NotImplementedError

    def backward(self, in_grad):
        raise NotImplementedError

class SquareLoss(MyCriterion):
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


class CrossEntropyLoss(MyCriterion):
    def forward(self, input, label):
        # input: m*output  label: m * output
        self.input = input
        self.label = label

        exp_sum = np.sum(np.exp(input))
        exp_pro = np.sum(np.exp(input * label))
        outputs = np.log(exp_sum) - np.log(exp_pro)
        outputs = -outputs / label.size
        return outputs

    def backward(self, in_grad):
        # return m * output
        # grad(log sum e^hat_y_i )
        out_grad = np.exp(self.input) / np.sum(np.exp(self.input))
        # -= grad(log sum e^(hat_y_i * y_i) )
        out_grad -= np.exp(self.input * self.label) / np.sum(np.exp(self.input * self.label))
        out_grad /= self.label.shape[1]
        return out_grad
