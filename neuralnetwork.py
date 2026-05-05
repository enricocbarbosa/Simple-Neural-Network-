from layer import Layer


class NeuralNetwork:
    # [1, 8, 4, 1]
    def __init__(self, layers:list):
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i], layers[i-1]))

    def forward(self):
        pass

    def backward(self):
        pass

    def train(self):
        pass