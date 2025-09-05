"""
Dense will inherit the layer abstract class
"""
from .layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size:int, output_size:int):
        """
        args: 
            input_size: number of neuron output in previous layer
            output_size: number of neuron you want in this layer
        """
        # for 4 input neurons, input_size is 4, and for 4 hidden layer neuron, outputsize = 4
        super().__init__()
        # Let's initilize the weights and bias
        self.weight = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        # print(self.weight)
        # print(self.bias)

    def forward_prop(self, input):
        # Let's save the input data
        self.input = input

        # Doing the matrix calculation
        return np.matmul(self.input, self.weight) + self.bias

    def backward_prop(self):
        pass
