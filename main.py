from neuralnetwork import NeuralNetwork


inputs = []
targets = []

for i in range(50):
    inputs.append([i])
    targets.append([i*2])

neural_network = NeuralNetwork([1, 8, 4, 1])
neural_network.train(inputs, targets, lr=0.0001)

testing_inputs = [[72], [4], [350]]
expected_outputs = [[144], [8], [700]]
for i in range(len(testing_inputs)):
    predicted_output = neural_network.forward(testing_inputs[i])
    print(f"Input: {testing_inputs[i]}. Expected: {expected_outputs[i]}. Predicted: {predicted_output}")
