from model import *
import pdb

net = pred_net(s_dim=2, a_dim=2, p_dim=2).float()
state = Variable(torch.rand(1,2)).float()
act_sequence = Variable(torch.rand(1,10,2)).float()
latent = Variable(torch.rand(1,2)).float()
preds, rewards = net(state, act_sequence, latent)
print(preds.size(), rewards.size())
