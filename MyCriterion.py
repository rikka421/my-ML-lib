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

        outputs = np.logaddexp.reduce(self.input, axis=1) - np.logaddexp.reduce(self.input * self.label, axis=1)
        outputs = np.sum(outputs) / label.size
        return outputs

    def backward(self):
        # return m * output
        # grad(log sum e^hat_y_i )
        out_grad = np.exp(self.input) * np.logaddexp.reduce(self.input, axis=1).reshape(self.input.shape[0], 1)
        # -= grad(log sum e^(hat_y_i * y_i) )
        out_grad -= np.exp(self.input * self.label) * np.logaddexp.reduce(self.input * self.label, axis=1).reshape(self.input.shape[0], 1)
        out_grad /= self.label.shape[1]
        return out_grad

if __name__ == '__main__':
    criterion = CrossEntropyLoss()

    Y = np.array([[0, 0, 1]])
    pre_Y = np.array([[1, 2, 3]])

    a = np.sum(np.exp(pre_Y * Y))
    b = np.sum(np.exp(pre_Y))
    c = - np.log(a / b) / 3
    print(a, b, c)
    print(criterion.forward(pre_Y, Y))

    d = np.log(b) * np.exp(pre_Y) - np.log(a) * np.exp(pre_Y * Y)
    e = d / 3

    print(d, e)

    print(criterion.backward())