import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 

from torchvision import transforms

import numpy as np
import pandas as pd

class NN(nn.Module):
    def __init__(self, batch_size, input_size):
        super(NN, self).__init__()
        h1 = nn.Linear(input_size, 100)
        h2 = nn.Linear(100, 35)
        h3 = nn.Linear(35, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.Tanh(),
            h2,
            nn.Tanh(),
            h3,
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)

    def predict(self, data):
        out = self.hidden(data)
        predict = [round(i.item()) for i in out]
        return predict