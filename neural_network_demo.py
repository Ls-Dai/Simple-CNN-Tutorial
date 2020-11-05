import torch 
import torch.nn as nn
from torch.nn.modules.module import Module 

class Hyperpara():
    def __init__(self) -> None:
        self.N = 64
        self.D_in = 1000
        self.H = 100
        self.D_out = 10

class Model(nn.Module):
    def __init__(self, hyperpara: Hyperpara):
        super(Model, self).__init__()
        N = hyperpara.N
        D_in = hyperpara.D_in
        H = hyperpara.H
        D_out = hyperpara.D_out
        
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

hyperpara = Hyperpara()

# Create random Tensors to hold inputs and outputs
x = torch.randn(hyperpara.N, hyperpara.D_in)
y = torch.randn(hyperpara.N, hyperpara.D_out)

# Construct our model by instantiating the class defined above
model = Model(hyperpara)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for it in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(loss.item())
