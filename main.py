from lib.activation import ReLU
from lib.activation import Sigmoid
from lib.dense import Dense
import numpy as np

X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1, 0, 1])

# First Layer with 5 neurons
layer1 = Dense(4, 5)
print("First layer created")
output = layer1.forward_prop(X)
print("First output without activation created")
# print("\nThis is the output:")
# print(output)
print(f"Output1 shape: {np.shape(output)}")
activation1 = ReLU()
print("Output1 now goes to activation")
y_pred = activation1.forward_prop(output)
print(f"shape of y_pred1 after activation: {np.shape(y_pred)}\n")

# Second Layer
# I will create another layer of 7 neurons
layer2 = Dense(5, 7)
print("layer2 created")
output2 = layer2.forward_prop(y_pred)
print("output2 created")
print(f"shape of output2 without activation: {np.shape(output2)}")

activation2 = ReLU()
print("activation2 created")
y_pred2 = activation2.forward_prop(output2)
print(f"shape of y_pred2{np.shape(y_pred2)}")

# Final layer (Output Layer)
layer3 = Dense(7, 1)
print("\nlayer3 created")
output3 = layer3.forward_prop(y_pred2)
print("output3 is generated")
print(f"\nShape of output3: {np.shape(output3)}")

activation3 = Sigmoid()
final_prediction = activation3.forward_prop(output3)
print(f"\n{final_prediction}")
print(f"\n shape of prediction after final activation{np.shape(final_prediction)}")