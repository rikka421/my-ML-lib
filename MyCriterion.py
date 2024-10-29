import numpy as np

class MyCriterion():
    def __init__(self):
        self.input = None
        self.label = None

    def forward(self, inputs, labels):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class SquareLoss(MyCriterion):
    def forward(self, input, label):
        # input: m*output  label: m * output
        self.input = input
        self.label = label
        # print(self.input.shape, self.label.shape)

        outputs = np.sum(np.square(input - label)) / (2 * label.shape[1])
        # print(np.max(input), np.max(label), np.max(outputs))
        return outputs

    def backward(self):
        # return m * output
        return (self.input - self.label) / (self.label.shape[1])


class CrossEntropyLoss(MyCriterion):
    def forward(self, input, label):
        # input: m*output  label: m * output
        self.input = input
        self.label = label

        outputs = (np.logaddexp.reduce(self.input, axis=1)
                   - np.sum(self.input * self.label, axis=1))
        return outputs

    def backward(self):
        # return m * output
        # grad(log sum e^hat_y_i )
        out_grad = np.exp(self.input) / np.sum(np.exp(self.input), axis=1)
        # -= grad(log sum e^(hat_y_i * y_i) )
        out_grad -= self.label
        return out_grad

if __name__ == '__main__':
    criterion = CrossEntropyLoss()

    Y = np.array([[0, 0, 1]])
    pre_Y = np.array([[1, 2, 3]])

    print(criterion.forward(pre_Y, Y))
    print(criterion.backward())