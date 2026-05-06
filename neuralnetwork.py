from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list):
        # layers = [1, 8, 4, 1] --> model
        # Creating all the layers passing the n of neurons and inputs
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i], layers[i-1]))

    def forward(self, inputs: list) -> list:
        # Passing the input for each layer getting its output, return the output of the last layer
        for layer in self.layers:
            layer_output = layer.forward(inputs)
            inputs = layer_output.copy()
        return inputs

    def backward(self, target: float, prediction: float, lr: float=0.001):
        gradients = [self.mse_gradient(target, prediction)]
        for layer in self.layers[::-1]:
            gradients = layer.backward(gradients, lr)

    @staticmethod
    def mse_gradient(target: float, prediction: float) -> float:
        return prediction - target

    @staticmethod
    def mse(target: float, prediction: float) -> float:
        return (prediction - target)**2

    def train(self, inputs: list, targets: list, epochs: int=200, lr: float=0.001):
        for i in range(epochs):
            total_error = 0
            for j in range(len(inputs)):
                output = self.forward(inputs[j])
                prediction = output[0]
                target = targets[j][0]
                total_error += self.mse(target, prediction)
                self.backward(target, prediction, lr)
            if i % 20 == 0:
                print(f"Epoch {i}: Loss = {total_error:.4f}")