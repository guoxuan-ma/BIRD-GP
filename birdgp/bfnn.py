import numpy as np
import torch

class BFNN(torch.nn.Module):
    def __init__(self, d, L, n, V):
        super().__init__()
        self.d = d
        self.L = L
        self.n = n
        self.V = V
        self.theta = torch.nn.Parameter(torch.randn(self.n, self.L))
        
        k = 256
        self.layer1 = torch.nn.Linear(d, k)
        self.layer2 = torch.nn.Linear(k, k)
        self.layer3 = torch.nn.Linear(k, k)
        self.layer4 = torch.nn.Linear(k, k)
        self.layer5 = torch.nn.Linear(k, k)
        self.layerL = torch.nn.Linear(k, L)

        self.relu = torch.nn.ReLU()

    def forward(self, X):
        X = self.layer1(X)
        X = self.relu(X)
        X = self.layer2(X)
        X = self.relu(X)
        X = self.layer3(X)
        X = self.relu(X)
        X = self.layer4(X)
        X = self.relu(X)
        X = self.layer5(X)
        X = self.relu(X)
        self.Psi = self.layerL(X)
        self.yhat = self.theta @ self.Psi.t() 
        return self.yhat
       
'''
class BFNN(torch.nn.Module):
    def __init__(self, d, L, n, V):
        super().__init__()
        self.d = d
        self.L = L
        self.n = n
        self.V = V
        self.theta = torch.nn.Parameter(torch.randn(self.n, self.L))
        
        self.layer1 = torch.nn.Linear(d, 4096)
        self.layer2 = torch.nn.Linear(4096, 256)
        self.layerL = torch.nn.Linear(256, L)

        self.relu = torch.nn.ReLU()

    def forward(self, X):
        X = self.layer1(X)
        X = self.relu(X)
        X = self.layer2(X)
        X = self.relu(X)
        self.Psi = self.layerL(X)
        self.yhat = self.theta @ self.Psi.t() 
        return self.yhat
'''