import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetLinear(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(FCNetLinear, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 20, bias=False)
        self.fc2 = nn.Linear(20, 20, bias=False)
        self.fc3 = nn.Linear(20, 20, bias=False)
        self.fc4 = nn.Linear(20, 20, bias=False)
        self.fc5 = nn.Linear(20, 20, bias=False)
        self.fc6 = nn.Linear(20, 1, bias=False)

    # self.dropout = nn.Dropout2d()  #Dropout

    def forward(self, x):
        # TODO no nonlinearity on last layer
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x
