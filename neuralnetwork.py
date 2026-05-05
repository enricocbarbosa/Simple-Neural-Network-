from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list):
        # layers = [1, 8, 4, 1]
        # Creating all the layers passing the n of neurons and inputs
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i], layers[i-1]))

    def forward(self, inputs: list):
        # Passing the input for each layer getting its output, return the output of the last layer
        for layer in self.layers:
            layer_output = layer.forward(inputs)
            inputs = layer_output.copy()
        return inputs

    def backward(self):
        pass

    def train(self):
        pass