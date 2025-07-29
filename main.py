import network
import matplotlib.pyplot as plt
import numpy

def getAnswerFromIndex(index: int):
    toReturn = []
    for i in range(10):
        toReturn.append(0.01)
    toReturn[index] = 0.99
    return toReturn

#constants
e = 2.71828

#settings
layer_neuron_count = [784, 200, 10]
activation_function = lambda x : 1 / (1 + e ** (-x))
learning_rate = 0.1
epochs = 5

#get training data
print("getting training data")
data_file = open("data/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

inputs = []
answers = []

for i in data_list:
    all_values = i.split(',')
    input = numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99 + 0.01
    answer = getAnswerFromIndex(int(all_values[0]))
    inputs.append(input)
    answers.append(answer)

#get test data
print("getting test data")
data_file = open("data/mnist_test.csv", 'r')
data_list = data_file.readlines()
data_file.close()

test_inputs = []
test_answers = []

for i in data_list:
    all_values = i.split(',')
    input = numpy.asarray(all_values[1:], dtype=float) / 255.0 * 0.99 + 0.01
    test_inputs.append(input)
    test_answers.append(int(all_values[0]))

#create neural network
neural_network = network.NeuralNetwork(layer_neuron_count, activation_function, learning_rate)

#train
print("Training:")
neural_network.train(inputs, answers, epochs)

#load weights
#neural_network.loadWeigths("save.w")

#test training
neural_network.test(test_inputs, test_answers)

#save weights
#neural_network.saveWeights("save.w")