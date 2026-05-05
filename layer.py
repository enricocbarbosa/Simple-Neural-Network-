from neuron import Neuron


class Layer:
    def __init__(self, n_neurons:int, n_inputs:int):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.neurons = []
        self.inputs = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def forward(self, inputs:list) -> list:
        # inputs = list with all the inputs
        # outputs = list with the output of each neuron
        self.inputs = inputs
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(self.inputs))

        return outputs

    def backward(self, gradients:list, lr:float=0.001) -> list:
        # gradients = list with the gradient for each neuron, came from the next layer
        # grad_w = list with other lists with a gradient for each weight of each neuron
        # grad_w_neuron = list with the gradient for each weight of a single neuron
        grad_w = []
        # Loop with the size of the amount of neurons
        for i in range(len(self.neurons)):
            grad_w_neuron = []
            # Loop with the size of the amount of inputs that is the same amount of weights
            for j in range(len(self.inputs)):
                # For each weight append to the list its gradient (gradient of the neuron * input)
                grad_w_neuron.append(gradients[i] * self.inputs[j])
            grad_w.append(grad_w_neuron)

        # grad_b = list with the gradient for each bias (the same as the neuron gradient)
        grad_b = gradients.copy()

        # Updating the parameters
        self.update(grad_w, grad_b, lr)

        # List with the gradient of each input
        grad_inputs = []
        for i in range(len(self.inputs)):
            grad_input = 0
            for j, neuron in enumerate(self.neurons):
                # Gradient of an input equals the weight for that input times the gradient of the neuron
                grad_input += neuron.w[i] * gradients[j]
            grad_inputs.append(grad_input)

        return grad_inputs

    def update(self, grad_w:list, grad_b:list, lr:float=0.001):
        # Loop to call the update for each neuron
        for i, neuron in enumerate(self.neurons):
            neuron.update(grad_w[i], grad_b[i], lr)
