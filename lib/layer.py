"""
I will create a sequential
"""

class Layer:
    def __init__(self):
        self.input = None
        
    def forward_prop(self, input_data):
        raise NotImplementedError

    def backward_prop(self):
        raise NotImplementedError