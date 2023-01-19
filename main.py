import numpy as np
import pandas as pd
import math
# Importing all activation functions
from activations import *
from lossfunctions import *
from accuracyfunctions import *

# "Neural Networks from Scratch in Python" by H Kinsley helped understand matrix dimensions for dot products
# The code commented below was adapted from this book specifically knowing to use the keepdims=True is code taken directly
# self.weight_derivs = np.dot(self.inputs.T, deriv_output)
# self.bias_derivs = np.sum(deriv_output, axis=0, keepdims=True)
# self.input_derivs = np.dot(deriv_output, self.weights.T)

# Snippet of code creates numpy arrays from our CSV dataset file
df = pd.read_csv('wdbc.data', header=None)
X = df.drop(columns=[0,1]).to_numpy()
Y_string = df[1].to_numpy()

# Below code converts labels to integer values were Malignant is 1
i = 0
Y = np.zeros(len(Y_string)).astype(int)
while i <= len(Y_string) - 1:
    if Y_string[i] == 'M':
        Y[i] = 1
    i += 1

# Class layer here
class Layer():

    def __init__(self, input_count, neuron_count):
        # Array of biases, one per node
        self.biases = np.zeros((1, neuron_count))
        # Matrix weights structured like this to avoid transposition in forward calculation
        # Reduced weights number here to try and avoid exploding gradients problem
        self.weights = np.random.rand(input_count, neuron_count) * 0.01
        self.layer_type = "layer"

    def forward(self, inputs):
        self.input = inputs
        # Output will be in shape (number of samples, number of outputs)
        self.output = np.dot(inputs, self.weights) + self.biases

    # "Neural Networks from Scratch in Python" by H Kinsley
    def backward(self, deriv_output):
        self.weight_derivs = np.dot(self.input.T, deriv_output)
        self.bias_derivs = np.sum(deriv_output, axis=0, keepdims=True)
        self.deriv_input = np.dot(deriv_output, self.weights.T)


# Below code builds an array containing each layer
def build_network(network_structure, activations):
    network = []
    i = 0
    while i <= len(network_structure) - 1:
        if i == 0:
            layer = Layer(X.shape[1], network_structure[i])
        else:
            layer = Layer(network_structure[i - 1], network_structure[i])
        network.append(layer)

        activ_layer = activations[i]
        network.append(activ_layer)
        i += 1
    return network

# Method iterates over the network calling the forward pass of each layer
# Then passing its outputs to the next
def forward_pass(network, x_batch):
    initial_layer = True
    previous_layer = None
    for layer in network:
        if initial_layer == True:
            layer.forward(x_batch)
            previous_layer = layer
            initial_layer = False
        else:
            layer.forward(previous_layer.output)
            previous_layer = layer


# Backward pass method iterates over a sigmoid activation output network in reverse
# First the loss functions deriv input is taken
# Partial derivatives are then calculated via passing the previous layers input derivative to the previous layer
# This follows the chain rule for calculating derivatives
def backward_pass(network, loss_function):
    loss = loss_function.forward(network[len(network) - 1].output)
    loss_function.backward()
    input_deriv = loss_function.deriv_input
    for layer in reversed(network):
        layer.backward(input_deriv)
        input_deriv = layer.deriv_input

# Same as the above backward_pass function
# As the derivative calculation for the softmax and loss function are combined we take our initial partial derivative
# Straight from the last layer of the network not the loss function which is combined into this layer
def backward_pass_conjoined_loss_activ(network):
    last_layer = True
    for layer in reversed(network):
        if last_layer:
            layer.backward(layer.output)
            input_deriv = layer.deriv_input
            last_layer = False
        else:
            layer.backward(input_deriv)
            input_deriv = layer.deriv_input

# This method simply lets users input hyperparameters of the model to train on the dataset
def gui():
    output_activation_choice = int(input("Output Activation Choice\n1: Sigmoid\n2: Softmax\nInput choice 1 or 2: "))
    number_of_hidden_layers_choice = int(input("Enter number of hidden layers: "))
    network_architecture = []
    activation_functions = []
    for layer in range(number_of_hidden_layers_choice):
        node_input_string = "Input number of nodes for hidden layer " + str(layer + 1) + " : "
        node_count_input = int(input(node_input_string))
        network_architecture.append(node_count_input)
        activation_input_string = "Input activation option hidden layer " + str(layer + 1) + " : "
        print("Options:\n1:Relu\n2:LeakyRelu\n3:Tanh\n4:Sigmoid")
        activation_input = int(input(activation_input_string))
        if activation_input == 1:
            activation = Relu()
        elif activation_input == 2:
            activation = LeakyRelu()
        elif activation_input == 3:
            activation = Tanh()
        elif activation_input == 4:
            activation = Sigmoid()
        activation_functions.append(activation)
    # Insert the output layer
    if output_activation_choice == 1:
        network_architecture.append(1)
        activation_functions.append(Sigmoid())
    elif output_activation_choice == 2:
        network_architecture.append(2)
        # Ensure this Y is replaced on training with the batched Y
        activation_functions.append(Softmax_CrossEntropyLossConjoined(Y))

    network = build_network(network_architecture, activation_functions)

    gradient_descent_option = int(input("1:stochastic gradient descent\n2:Mini-Batch Gradient Descent"
                                        "\n3:Batch Gradient Descent\nInput gradient descent option: "))
    if gradient_descent_option == 1:
        batch_size = 1
    elif gradient_descent_option == 2:
        batch_size_input = int(input("Batch size: "))
        batch_size = batch_size_input
    elif gradient_descent_option == 3:
        batch_size = len(Y)
    return network, batch_size, output_activation_choice


# Additional method allowing users to input hyperparamaters for learning rate scheduling
def gui_hyperparams():
    lr = float(input("Input learning rate: "))
    epochs = int(input("Input epochs: "))
    lr_schedule = int(input("1: Default\n2: Learning rate with decay\n3: Drop based learning rate"
                            "\n4: Learning rate with momentum\n5: Learning rate with momentum and decay\n"))
    lr_decay = 0.
    lr_drop_freq = 0
    lr_drop_rate = 0
    momentum = 0

    if lr_schedule == 2:
        lr_decay = float(input("Learning rate decay: "))
    if lr_schedule == 3:
        lr_drop_freq = float(input("Learning rate drop frequency: "))
        lr_drop_rate = float(input("Learning rate drop rate: "))
    if lr_schedule == 4:
        momentum = float(input("Learning rate momentum: "))
    if lr_schedule == 5:
        lr_decay = float(input("Learning rate decay: "))
        momentum = float(input("Learning rate momentum: "))

    return lr, epochs, lr_schedule, lr_decay, lr_drop_freq, lr_drop_rate, momentum

# Method take the X and Y numpy arrays produced from the CSV dataset and a user defined batch size
# The numpy arrays are split into multiple numpy arrays the size of the batch size
# These arrays are then stored in a list and returned
def create_batches(batch_size, X, Y):
    # Calculate the number of steps per epoch
    step_count = len(Y) // batch_size

    if len(Y) % batch_size != 0:
        step_count += 1

    index_for_iloc = batch_size
    x_batched = []
    y_batched = []
    for step in range(step_count):
        x_batch = X[(index_for_iloc - batch_size):index_for_iloc]
        y_batch = Y[(index_for_iloc - batch_size):index_for_iloc]
        x_batched.append(x_batch)
        y_batched.append(y_batch)
        index_for_iloc += batch_size
    return x_batched, y_batched, step_count


# Updating weights and biases class is instantiated pre-training of a model
# Handles the learning rate schedule and updating each layers weights and biases with the calculated partial derivs
class update_weights_biases:

    def __init__(self, network, epoch, lr, schedule, decay=0., drop_rate=0., drop_freq=0, momentum = 0):
        self.network = network
        self.epoch = epoch
        self.lr = lr
        self.initial_lr = lr
        self.schedule = schedule
        self.decay = decay
        self.drop_rate = drop_rate
        self.drop_freq = drop_freq
        self.momentum = momentum

    def update(self):
        if self.schedule == "default":
            for layer in self.network:
                if layer.layer_type == "layer":
                    layer.weights = layer.weights - (layer.weight_derivs * self.lr)
                    layer.biases = layer.biases - (layer.bias_derivs * self.lr)
        if self.schedule == "decay":
            for layer in self.network:
                if layer.layer_type == "layer":
                    self.lr = self.lr * 1 / (1 + self.decay * self.epoch)
                    layer.weights = layer.weights - (layer.weight_derivs * self.lr)
                    layer.biases = layer.biases - (layer.bias_derivs * self.lr)
        if self.schedule == "drop-based":
            for layer in self.network:
                if layer.layer_type == "layer":
                    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
                    # Code was converted to python from code found in the site above
                    self.lr = self.initial_lr * math.pow(self.drop_rate,
                                                         math.floor((1+self.epoch)/self.drop_freq))
                    layer.weights = layer.weights - (layer.weight_derivs * self.lr)
                    layer.biases = layer.biases - (layer.bias_derivs * self.lr)
        if self.schedule == "momentum":
            for layer in self.network:
                if layer.layer_type == "layer":
                    # "Neural Networks from Scratch in Python" by H
                    # Momentum code taken from book
                    if not hasattr(layer, "weight_momentums"):
                        layer.weight_momentums = np.zeros_like(layer.weights)
                        layer.bias_momentums = np.zeros_like(layer.biases)
                    weight_updates = self.momentum * layer.weight_momentums - self.lr * layer.weight_derivs
                    layer.weight_momentums = weight_updates
                    layer.weights = layer.weights + weight_updates
                    bias_updates = self.momentum * layer.bias_momentums - self.lr * layer.bias_derivs
                    layer.bias_momentums = bias_updates
                    layer.biases = layer.biases + bias_updates
        if self.schedule == "momentum_with_decay":
            for layer in self.network:
                if layer.layer_type == "layer":
                    # "Neural Networks from Scratch in Python" by H Kinsley
                    # Momentum code taken from book
                    if not hasattr(layer, "weight_momentums"):
                        layer.weight_momentums = np.zeros_like(layer.weights)
                        layer.bias_momentums = np.zeros_like(layer.biases)
                    self.lr = self.lr * 1 / (1 + self.decay * self.epoch)
                    weight_updates = self.momentum * layer.weight_momentums - self.lr * layer.weight_derivs
                    layer.weight_momentums = weight_updates
                    layer.weights = layer.weights + weight_updates
                    bias_updates = self.momentum * layer.bias_momentums - self.lr * layer.bias_derivs
                    layer.bias_momentums = bias_updates
                    layer.biases = layer.biases + bias_updates

# Method responsible for running the epochs of a model with a softmax output
# Takes in the models hyperparameters
# Returns a dictionary containing saved information on the models performance each epoch
def train_softmax_batches(network, step_count, x_batches, y_batches, lr, epochs, lr_schedule,
                          lr_decay, drop_freq, drop_rate, momentum):
    if lr_schedule == 1:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="default")
    if lr_schedule == 2:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="decay", decay=lr_decay)
    if lr_schedule == 3:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="drop-based",
                                               drop_rate=drop_rate, drop_freq=drop_freq)
    if lr_schedule == 4:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="momentum",
                                               momentum=momentum)
    if lr_schedule == 5:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="momentum_with_decay",
                                               momentum=momentum, decay=lr_decay)
    epoch_list = []
    lr_list = []
    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        update_network.epoch = epoch
        acc_loss = 0
        acc_acc = 0
        for step in range(step_count):
            acc_function = SoftmaxAccuracy(y_batches[step])
            network[len(network) - 1].Y = y_batches[step]
            forward_pass(network, x_batches[step])
            backward_pass_conjoined_loss_activ(network)
            update_network.update()
            loss = network[len(network) - 1].loss
            acc = acc_function.calculate(network[len(network) - 1].output)
            acc_loss += loss
            acc_acc += acc
        loss = acc_loss / step_count
        accuracy = acc_acc / step_count
        print("Loss: ", loss, "Acc: ", accuracy, "Epoch: ", epoch, "Lr: ", update_network.lr)
        epoch_list.append(epoch)
        lr_list.append(update_network.lr)
        acc_list.append(accuracy)
        loss_list.append(loss)
    network_info = {
        "epoch" : epoch_list,
        "lr" : lr_list,
        "acc": acc_list,
        "loss": loss_list
    }
    print(network_info)
    return network_info


# Method responsible for running the epochs of a model with a softmax output
# Takes in the models hyperparameters
# Returns a dictionary containing saved information on the models performance each epoch
def train_sigmoid_batches(network, step_count, x_batches, y_batches, lr, epochs, lr_schedule,
                          lr_decay, drop_freq, drop_rate, momentum):
    if lr_schedule == 1:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="default")
    if lr_schedule == 2:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="decay", decay=lr_decay)
    if lr_schedule == 3:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="drop-based",
                                               drop_rate=drop_rate, drop_freq=drop_freq)
    if lr_schedule == 4:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="momentum",
                                               momentum=momentum)
    if lr_schedule == 5:
        update_network = update_weights_biases(network=network, epoch=0, lr=lr, schedule="momentum_with_decay",
                                               momentum=momentum, decay=lr_decay)
    loss_function = BinaryCrossEntropyLoss(y_batches[0])
    epoch_list = []
    lr_list = []
    acc_list = []
    loss_list = []
    for epoch in range(epochs):
        update_network.epoch = epoch
        acc_loss = 0
        acc_acc = 0
        for step in range(step_count):
            # Update with batched y values
            acc_function = SigmoidAccuracy(y_batches[step])
            loss_function.y_true = y_batches[step].reshape(-1, 1)

            forward_pass(network, x_batches[step])
            backward_pass(network, loss_function)

            update_network.update()

            loss = loss_function.loss
            acc = acc_function.calculate(network[len(network) -1].output)

            acc_loss += loss
            acc_acc += acc
        loss = acc_loss / step_count
        accuracy = acc_acc / step_count
        print("Loss: ", loss, "Acc: ", accuracy, "Epoch: ", epoch)
        epoch_list.append(epoch)
        lr_list.append(update_network.lr)
        acc_list.append(accuracy)
        loss_list.append(loss)
    network_info = {
        "epoch": epoch_list,
        "lr": lr_list,
        "acc": acc_list,
        "loss": loss_list
    }
    print(network_info)
    return network_info

# Code within this "if statement" only ran if this script is ran specifically instead of running when imported from
if __name__ == "__main__":
    network, batch_size, activation_choice = gui()
    lr, epochs, lr_schedule, lr_decay, lr_drop_freq, lr_drop_rate, momentum = gui_hyperparams()

    x_batches, y_batches, step_count = create_batches(batch_size, X, Y)
    print(y_batches[0].shape)

    if activation_choice == 1:
        train_sigmoid_batches(network, step_count, x_batches, y_batches, lr, epochs, lr_schedule, lr_decay,
                              lr_drop_freq,
                              lr_drop_rate, momentum)
    elif activation_choice == 2:
        train_softmax_batches(network, step_count, x_batches, y_batches, lr, epochs, lr_schedule, lr_decay,
                              lr_drop_freq,
                              lr_drop_rate, momentum)

