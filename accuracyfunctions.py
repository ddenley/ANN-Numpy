import numpy as np

# Sigmoid accuracy works via first converting the output to a binary value
# Then we calculate the mean of how many times this binary value is equal to the ground truth
class SigmoidAccuracy:

    def __init__(self, Y, boundary_for_NM = 0.5):
        self.Y = Y
        self.boundary_for_NM = boundary_for_NM

    def calculate(self, output):
        i = 0
        while i < len(output):
            if output[i][0] <= self.boundary_for_NM:
                output[i][0] = 0
            else:
                output[i][0] = 1
            i += 1
        acc = np.mean(output[: , 0] == self.Y)
        return acc


# Softmax accuracy iterates through the output and compares to the ground thruth label
class SoftmaxAccuracy:

    def __init__(self, Y):
        self.Y = Y

    def calculate(self, output):
        i = 0
        correct_predictions = 0
        while i < len(self.Y):
            if np.argmax(output[i]) == self.Y[i]:
                correct_predictions += 1
            i += 1
        acc = correct_predictions / len(self.Y)
        return acc