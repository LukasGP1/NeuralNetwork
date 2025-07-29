import sys
from typing import List
from typing import Callable
import numpy

def update_progress_bar(percentage, bar_length=40):
    completed = int(bar_length * (percentage / 100))
    bar = '#' * completed + '-' * (bar_length - completed)
    print(' ' * (bar_length + 10), end='\r')  # Clear the line
    print(f'[{bar}] {percentage:.2f}%', end='\r', flush=True)

def max_index(list: List[float]) -> int:
    max_index = 0
    for i in range(len(list)):
        if list[i] > list[max_index]:
            max_index = i
    return max_index

class NeuralNetwork:
    def __init__(self, layer_neuron_count: List[int], activation_function: Callable[[float], float], alpha: float):
        self.activation_function = activation_function
        self.alpha = alpha

        if len(layer_neuron_count) < 2:
            sys.exit("Fatal error: Neural Network has to have at least two layers")
        for i in layer_neuron_count:
            if i < 1:
                sys.exit("Fatal error: Neural Network layer has to have at least one neuron")
        self.layer_neuron_count = layer_neuron_count
        self.layers = len(layer_neuron_count)

        #initialize weights
        self.layers_weights = []
        for i in range(self.layers - 1):
            self.layers_weights.append(numpy.random.normal(0.0, pow(layer_neuron_count[i], -0.5), (layer_neuron_count[i + 1], layer_neuron_count[i])))
        
        self.layers_weights_copy = self.layers_weights.copy()

        parameters = 0
        for i in range(self.layers - 1):
            parameters += self.layer_neuron_count[i] * self.layer_neuron_count[i + 1]

        print("Created Neural Network with " + str(parameters) + " parameters")
        pass

    def query(self, input: List[float]) -> List[float]:
        if len(input) != self.layer_neuron_count[0]:
            sys.exit("Fatal error: Input has to have the same size as first layer of neural network")

        #put input in correct format
        input = numpy.transpose(numpy.atleast_2d(input))

        #traverse through neural network
        next_layer_input = input.copy()
        for w in self.layers_weights:
            next_layer_input = self.activation_function(w @ next_layer_input)
        
        return next_layer_input

    def test(self, inputs: List[List[float]], answers: List[int]):
        if len(inputs) != len(answers):
            sys.exit("Fatal error: Answers has to have the same size as Inputs")
        if len(inputs) < 1:
            sys.exit("Fatal error: You have to specify at least one testing case")
        for answer in answers:
            if answer < 0 or answer > self.layer_neuron_count[-1] - 1:
                sys.exit("Fatal error: Invalid answer")
        
        correct = 0
        for i in range(len(inputs)):
            input = inputs[i]
            answer = answers[i]
            output = self.query(input)
            if max_index(output) == answer:
                correct += 1
        
        percentage = correct / len(inputs) * 100
        print("Accuracy: " + str(percentage) + "%")

    def trainOneExample(self, input: List[float], answer: List[float]):
        if len(input) != self.layer_neuron_count[0]:
            sys.exit("Fatal error: Input has to have the same size as first layer of neural network")
        if len(answer) != self.layer_neuron_count[-1]:
            sys.exit("Fatal error: Answer has to have the same size as last layer of neural network")

        #put input and answer in the right format
        input = numpy.transpose(numpy.atleast_2d(input))
        answer = numpy.transpose(numpy.atleast_2d(answer))

        #query
        outputs = []
        next_layer_input = input.copy()
        for w in self.layers_weights:
            outputs.append(next_layer_input)
            next_layer_input = self.activation_function(w @ next_layer_input)
        outputs.append(next_layer_input)
        
        #calculate errors with backpropagation
        errors = [answer - outputs[-1]]
        for i in range(self.layers - 2):
            w = self.layers_weights[-i -1]
            errors.append(numpy.dot(w.T, errors[i]))

        #update weights with gradient descent
        for i in range(len(self.layers_weights)):
            self.layers_weights_copy[-i - 1] += self.alpha * numpy.dot((errors[i] * outputs[-i - 1] * (1 - outputs[-i - 1])), numpy.transpose(outputs[-i - 2]))
    
    def train(self, inputs: List[List[float]], answers: List[List[float]], epochs: int):
        if len(inputs) != len(answers):
            sys.exit("Fatal error: Answers has to have the same size as Inputs")
        if len(inputs) < 1:
            sys.exit("Fatal error: You have to specify at least one training example")
        
        for e in range(epochs):
            for i in range(len(inputs)):
                input = inputs[i]
                answer = answers[i]
                self.trainOneExample(input, answer)
                #percentage = ((e * len(inputs) + i) / (epochs * len(inputs))) * 100
                #update_progress_bar(percentage)
            self.layers_weights = self.layers_weights_copy.copy()
        print()
    
    def saveWeights(self, filePath: str):
        file = open(filePath, 'w')
        for w in self.layers_weights:
            for end in w:
                for start in end:
                    file.write(str(start) + "\n")
        file.close()
        print("Weights have been saved to " + filePath)
    
    def loadWeigths(self, filePath: str):
        file = open(filePath, 'r')
        for w in range(len(self.layers_weights)):
            for end in range(len(self.layers_weights[w])):
                for start in range(len(self.layers_weights[w][end])):
                    self.layers_weights[w][end][start] = float(file.readline())
        file.close()
        print("Weights have been loaded from " + filePath)