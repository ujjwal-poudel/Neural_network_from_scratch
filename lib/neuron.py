import numpy as np
import activation

class neuron:
    def __init__(self, input_array, weights, bias):
        """
        Shape of input_array: numpy array, (,n)
        Shape of weights: numpy array, (n, )
        """
        self.input_array = input_array
        self.weights = weights
        self.bias = bias

    def activate(self, activation_type: str) -> float:
        """
        The neuron will calculate the weighted sum and call activation function

        args:
            activation_type: The string of the type of activation function

        ouput:
            z: a scalar value after going through activation function
        """

        weighted_sum = np.matmul(self.input_array, self.weights) + self.bias

        if activation_type == 'relu':
            return activation.relu(weighted_sum[0][0])

        if activation_type == 'sigmoid':
            return activation.sigmoid(weighted_sum[0][0])