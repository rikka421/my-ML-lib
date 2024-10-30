import numpy as np

class Optimizer():
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        raise NotImplementedError

    def step(self, gradients):
        raise NotImplementedError

class SimpleSGD(Optimizer):
    def step(self, gradients):
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            param -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(params, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param) for param in params]   # 一阶矩估计
        self.v = [np.zeros_like(param) for param in params]   # 二阶矩估计
        self.t = 0                                # 计步

    def step(self, gradients):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # 更新一阶和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 参数更新
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


if __name__ == '__main__':
    params = [np.ones((3, 1))]
    gradients = [np.ones((3, 1))]
    adam = Adam(params)
    while adam.t < 100:
        adam.step(gradients)
        print(params, adam.m, np.sqrt(adam.v))
    """
    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数
    """