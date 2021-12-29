import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Q_LNN(nn.Module):
    def __init__(self, input_size = 11, output_size = 3, hidden_size = [256], dropout = 0):
        super().__init__()
        #Input layer
        input_layer = []
        input_layer.append(nn.Linear(input_size, hidden_size[0]))
        input_layer.append(nn.ReLU())
        #input_layer.append(nn.Dropout(dropout))
        self.input_layer = nn.Sequential(*input_layer)

        #Hidden layers in case we have more than one hidden layer
        self.has_hidden_layers = False
        if len(hidden_size) > 1:
            self.has_hidden_layers = True
            hidden_layers = []
            for i in range(len(hidden_size)-1):
                hidden_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
                hidden_layers.append(nn.ReLU())
                hidden_layers.append(nn.Dropout(dropout))
            self.hidden_layers = nn.Sequential(*hidden_layers)

        #Output layer
        output_layer = []
        output_layer.append(nn.Linear(hidden_size[-1], output_size))
        #output_layer.append(nn.Softmax())
        self.output_layer = nn.Sequential(*output_layer)

    def forward(self, x):
        x = self.input_layer(x)
        if self.has_hidden_layers:
            x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x



class Q_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 256)
        self.output = nn.Linear(256, 3)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x