"""
This module will contain the activation functions
"""
import numpy as np
from .layer import Layer

# It looks like I need to inherit the layer abstraction class to make it modular

class ReLU(Layer):

    def __init__(self):
        super().__init__()
        print("Activation Layer with ReLU created")

    def forward_prop(self, input):
        # Updating the input data
        self.input = input
        return np.where(input > 0, input, 0)
    
    def backward_prop(self):
        pass

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        print("Activation Layer with Sigmoid Created")

    def forward_prop(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))
