import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter()

x=torch.arange(-5,5,0.1).view(-1,1)
y=-5*x+0.1*torch.rand(x.size())

model=nn.Linear(1,1)
criterion=nn.MSELoss()
optimizer=torch.optim.AdamW(model.parameters())

def train(iter):
    for epoch in range(iter):
        y1=model(x)
        loss=criterion(y1,y)
        print(type(loss))
        writer.add_scalar("Loss/train",loss,epoch)
        writer.add_scalar("Loss/train2",loss,epoch)
        loss.backward()
        optimizer.step()

train(10)
print('over')
writer.flush()

