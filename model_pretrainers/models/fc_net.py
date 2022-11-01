import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(14 * 14, 30, bias=False)
        self.fc2 = nn.Linear(30, 30, bias=False)
        self.fc3 = nn.Linear(30, 30, bias=False)
        self.fc4 = nn.Linear(30, 30, bias=False)
        self.fc5 = nn.Linear(30, 10, bias=False)

    # self.dropout = nn.Dropout2d()  #Dropout

    def forward(self, x):
        # TODO no nonlinearity on last layer
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.dropout(x, training=self.training)
        return F.sigmoid(x)
        # return F.log_softmax(x, dim=1)
