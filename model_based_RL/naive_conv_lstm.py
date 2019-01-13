import torch
import torch.nn as nn
from nn_module import conv1d_block, conv2d_block, fc_block


class FCLSTMCell(nn.Module):

    def __init__(self, action_channels, state_channels, dynamics_channels,
                 init_type="xavier", use_batchnorm=False):
        super(FCLSTMCell, self).__init__()
        self.in_channels = action_channels + state_channels + dynamics_channels
        self.feature_channels = state_channels
        self.out_channels = state_channels * 4
        self.mid_channels = 32
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.fc1 = fc_block(in_channels=self.in_channels,
                            out_channels=self.mid_channels,
                            init_type=init_type,
                            activation=self.tanh,
                            use_batchnorm=use_batchnorm)
        self.fc2 = fc_block(in_channels=self.mid_channels,
                            out_channels=self.out_channels,
                            init_type=init_type,
                            activation=None,
                            use_batchnorm=use_batchnorm)
        self.reward_predict = fc_block(in_channels=self.in_channels + state_channels,
                                       out_channels=1,
                                       init_type=init_type,
                                       activation=self.tanh,
                                       use_batchnorm=use_batchnorm)

    def forward(self, hidden_states, action, dynamics):
        prev_h, prev_c = hidden_states
        assert(dynamics.size()[0] == action.size()[0])
        concat_feature = torch.cat([dynamics, action, prev_h], dim=1)
        fc_output1 = self.fc1(concat_feature)
        fc_output2 = self.fc2(fc_output1)
        act_i, act_f, hat_c, act_o = torch.split(fc_output2, self.feature_channels, dim=1)

        act_i = self.sigmoid(act_i)
        act_f = self.sigmoid(act_f)
        hat_c = self.tanh(hat_c)
        act_o = self.sigmoid(act_o)

        next_c = act_f * prev_c + act_i * hat_c
        next_h = act_o * self.tanh(next_c)

        concat_feature = torch.cat([concat_feature, next_h], dim=1)
        reward = self.reward_predict(concat_feature)

        return next_h, next_c, reward


class FCLSTM(nn.Module):

    def __init__(self, action_channels, state_channels, dynamics_channels, history_length):
        self.history_length = history_length
        self.state_channels = state_channels
        self.lstm = []
        for num in range(self.history_length):
            self.lstm.append(FCLSTMCell(action_channels, state_channels, dynamics_channels))

    def forward(self, dynamics, samples):
        assert(samples.size()[1] == self.history_length)
        b, h, w = samples.shape
        h = torch.zeros(b, self.state_channels).cuda()
        c = torch.zeros(b, self.state_channels).cuda()
        for num in range(self.history_length):
            h, c, reward = self.lstm[num]((h, c), samples[num], dynamics)

        return h, reward


if __name__ == "__main__":
    history_length = 2
    batch_size = 4
    action_channels = 8
    state_channels = 16
    dynamics_channels = 3
    samples = torch.randn(batch_size, history_length, action_channels).cuda()
    dynamics = torch.randn(batch_size, dynamics_channels).cuda()
    model = FCLSTM(action_channels, state_channels, dynamics_channels, history_length).cuda()
    h, reward = model(dynamics, samples)
    print(h.shape)
    print(reward.shape)
