import numpy as np

# Used when sigmoid is output layer
# Binary cross entropy loss formula:
# -y * log(y_pred) - (1 - y) * log(1-y_pred)
class BinaryCrossEntropyLoss:

    def __init__(self, y_true):
        # For calculating cross binary loss the y has to be reshaped
        self.y_true = y_true.reshape(-1, 1)

    def binarylossformula(self):
        return -self.y_true * np.log(self.y_pred) - (1 - self.y_true) * np.log(1 - self.y_pred)

    def forward(self, y_pred):
        self.y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss_over_all_samples = self.binarylossformula()
        # Calculate the mean loss for each sample - in SGD we will use batch sizes of 1 and this will still work
        loss_mean = np.mean(loss_over_all_samples)
        self.loss = loss_mean
        return loss_mean

    def backward(self):
        # Calculate gradient of each loss value
        # Formula: -( (y_true / y_pred) - ( (1 - y_true) / (1 - y_pred) ) )
        self.deriv_input = -( (self.y_true / self.y_pred) - ( (1 - self.y_true) / (1 - self.y_pred) ) )
        # Calculate the mean gradient of loss
        self.deriv_input = self.deriv_input / len(self.y_pred)


