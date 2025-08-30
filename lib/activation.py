"""
This module will contain the activation functions
"""
import numpy as np
# Rectified Linear Unit (ReLU)

def relu(x: float) -> float:
    try:
        if x > 0:
            return x
        else:
            return 0
    except Exception as error:
        print("Looks like you're tyring to enter a string\n")
        print(error)

def sigmoid(x: float) -> float:
    return 1 / (1 + (np.exp(-x)))

def softmax():
    # I'll create it later if I need it
    pass