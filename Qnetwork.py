

import torch.nn as nn
import torch as T
import torch.nn.functional as F

class QNetwork(nn.Module):
  def __init__(self,input_dim,n_action):
    super(QNetwork,self).__init__()
    self.f1=nn.Linear(input_dim,128)
    self.f2=nn.Linear(128,64)
    self.f3=nn.Linear(64,32)
    self.f4=nn.Linear(32,n_action)
    #self.optimizer=optim.Adam(self.parameters(),lr=lr)
    #self.loss=nn.MSELoss()
    self.device=T.device('cuda' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self,x):
    x=F.relu(self.f1(x))
    x=F.relu(self.f2(x))
    x=F.relu(self.f3(x))
    x=self.f4(x)
    return x

  def act(self,obs):
    #state=T.tensor(obs).to(device)
    state=obs.to(self.device)
    actions=self.forward(state)
    action=T.argmax(actions).item()

    return action