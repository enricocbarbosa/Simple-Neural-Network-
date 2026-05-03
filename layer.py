from neuron import Neuron


class Layer:
    def __init__(self, n_neurons:int, n_inputs:int):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def forward(self, inputs:list) -> list:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))

        return outputs

    def backward(self):
        pass

    def update(self):
        pass
