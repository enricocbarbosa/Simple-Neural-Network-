import random


class Neuron:
    def __init__(self, n_x:int):
        # w = list of weights, initially random, length passed as parameter (n_x)
        self.w = []
        for i in range(n_x):
            self.w.append(random.uniform(-1, 1))
        self.b = 0

    def forward(self, x:list) -> float:
        # z = sum of the products of each input and its respective weight
        z = 0
        for i in range(len(self.w)):
            z += (x[i] * self.w[i])
        z += self.b

        #ReLU (break linearity)
        if z < 0:
            z = 0

        return z

    def update(self, grad_w:list, grad_b:float, lr:float=0.001):
        # Updating the weights
        for i in range(len(self.w)):
            self.w[i] -= grad_w[i] * lr

        # Updating the bias
        self.b -= grad_b * lr

