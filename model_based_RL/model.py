import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=self.bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=self.bias)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        combined_conv = F.relu(self.W(x))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class pred_net(nn.Module):
    def __init__(self, s_dim, a_dim, p_dim):
        super(pred_net, self).__init__()
        self.fc = nn.Linear(s_dim + a_dim + p_dim, s_dim)
        self.lstm = LSTMCell(s_dim, s_dim, bias=False)
        self.reward_layer = nn.Linear(s_dim, 1)

    def forward(self, state, act_sequence, latent):
        batch_size = int(state.size()[0])
        hidden_state = torch.cat([state, act_sequence[:, 0, :], latent], dim=1)
        hidden = F.relu(self.fc(hidden_state))
        cell = F.relu(self.fc(hidden_state))
        pred_step = act_sequence.size(1)
        rewards = []
        preds = []
        for i in range(pred_step):
            hidden, cell = self.lstm(hidden, [hidden, cell])
            reward = self.reward_layer(hidden)
            rewards.append(reward)
            preds.append(hidden)
            if i < pred_step-1:
                hidden_state = torch.cat([hidden, act_sequence[:, i, :], latent], dim=1)
                hidden = F.relu(self.fc(hidden_state))
        return torch.cat(preds, dim=0), torch.cat(rewards, dim=0)
