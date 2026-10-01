import torch
import torch.nn as nn

# Feedforward Neural Networks
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)    # computer our linear function
                                                        # 1st hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)   # 2nd hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes)   # 3rd hidden layer
        self.relu = nn.ReLU()   # ReLu activation function
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #  no activation and no softmax
        return out
        