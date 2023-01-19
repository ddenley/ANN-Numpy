import numpy as np

# From the book "Neural Networks from Scratch in Python" by H Kinsley activation functions are treated almost as their own layers
# The code structure is inspired from this idea
# All forward and backward pass calculations are my own code implementations from mathematical formulas
# These formulas are commented above the numpy code implementation
# Where numpy solutions were taken the link is provided above the code

# Forward pass:
# Takes input from previous layer and applies the activation function to produce self.output
# self.output is saved to the class for later use in backward pass

# Sigmoid can be used in output layer for binary classification with one output node
class Sigmoid:

    def __init__(self):
        self.layer_type = "activation"

    def forward(self, input):
        # Save input
        self.input = input
        # Calculate forward pass, Logistic function = 1 / (1 + e^-x)
        logistic_function = 1 / (1 + np.exp(-input))
        # Save output for easier access in next layer and for backward pass calculation
        self.output = logistic_function

    def backward(self, deriv_output):
        # Deriv logistic function = f(x) * (1 - f(x))
        fx = self.output
        logistic_function_deriv = fx * (1 - fx) * deriv_output
        #chain_rule_applied = logistic_function_deriv * deriv_output
        self.deriv_input = logistic_function_deriv


class Relu:

    def __init__(self):
        self.layer_type = "activation"

    def forward(self, input):
        # Save input for later use
        self.input = input
        # Relu function: if x > 0 linear else 0
        self.output = np.maximum(0, input)

    def backward(self, deriv_output):
        # By looking at the Relu function graph we can see that the gradient of x < 0 is 0
        # The gradient if x > 0 is 1, chain rule applied this is simply 1 * deriv_output
        # So take the deriv_ouput matrix and zero the gradient were x < 0 and leave other derivs untouched
        self.deriv_input = deriv_output.copy()
        # Learned how to zero numpy values given a condition from the below stackoverflow question
        # https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-above-a-specific-threshold
        # Similar solution is also found in the book mentioned at the top of this script
        self.deriv_input[self.input <= 0] = 0

class LeakyRelu:

    def __init__(self, alpha=0.1):
        self.layer_type = "activation"
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        # Adapted from https://stackoverflow.com/questions/50517545/how-do-i-implement-leaky-relu-using-numpy-functions
        # Simply altered to work with my self.alpha value
        self.output = np.where(input > 0, input, input * self.alpha)

    def backward(self, deriv_output):
        # Very similiar to backward of Relu however if x < 0 gradient will be alpha value
        # As it is not zero the chain rule will also be applied to the alpha value
        # My soloution was to create an array of ones the same shape as deriv_output
        # As gradient is one where x > 0
        # Then instead of a zero gradient for x < 0 set gradient to alpha (my set gradient)
        # Finally apply chain rule
        self.deriv_input = np.ones_like(deriv_output)
        self.deriv_input[self.input <= 0] = self.alpha
        self.deriv_input = self.deriv_input * deriv_output

class Tanh:

    def __init__(self):
        self.layer_type = "activation"

    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)

    def backward(self, deriv_output):
        self.deriv_input = (1 - (np.tanh(self.input) ** 2)) * deriv_output


# Activation function and Loss function conjoined decision was made after reading the following article:
# https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
# Specifically it mentions the ease of backpropagation implementation and I am also aware of computation speed
# Note no code was taken from this and if code is taken it will be commented above the line within this function
class Softmax_CrossEntropyLossConjoined:

    def __init__(self, Y):
        self.layer_type = "activation_loss"
        self.Y = Y

    def forward(self, input):
        self.input = input
        # Formula is the exponetial of input divided by the sum of the exponentials
        # Soloution to this, specifically knowing how to use keepdims=True
        # Came from page 122 of "Neural Networks from Scratch Python"
        self.output = np.exp(self.input) / np.sum(np.exp(self.input), axis=1, keepdims=True)
        # Now for calculating the loss
        # Categorical Cross Entropy Loss is the negative log of the confidence of the correct class
        # We need to ensure we don't take a log of 0 or 1 here, clip with very small values here
        output_clipped = np.clip(self.output, 1e-10, 1 - 1e-10)
        # Binary cross entropy loss can just be seen as the sum of negative logs of the correct confidence
        # In many tutorials they use one-hot encoded labels which is the method I will implement here
        # The idea behind this is it allows me to use the argmax function to find the correct confidences index
        Y_one_hot = np.zeros_like(output_clipped)
        i = 0
        while i < len(self.Y):
            if self.Y[i] == 1:
                Y_one_hot[i][1] = 1
            else:
                Y_one_hot[i][0] = 1
            i += 1
        # Now for negative log of the correct confidence score
        i = 0
        loss = 0
        while i < len(self.Y):
            confidence_index = np.argmax(Y_one_hot[i])
            loss += -np.log(output_clipped[i][confidence_index])
            i += 1
        self.loss = loss / len(self.Y)


    def backward(self, deriv_output):
        # I don't understand how the formula simplifies to this derivative equation but the combined deriv of
        # Cross-Entropy loss and softmax solves to : y_pred - y_actual
        # This equation is found on pg 229 of "Neural Networks from Scratch in Python"
        # NOTE: THE SOLUTION BELOW IS NOT MY CODE AND FROM PAGE 229 of the book mentioned - v slightly altered
        # to work with my dataset/variables
        self.deriv_input = deriv_output.copy()
        self.deriv_input[range(len(self.Y)), self.Y] -= 1
        self.deriv_input = self.deriv_input / len(self.Y)









