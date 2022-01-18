import torch.nn as nn
import torch.nn.functional as F
import torch
class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(6 * 6 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)

        x = F.relu(self.conv2(x))
        #print(x.shape)

        x = F.relu(self.conv3(x))
        #print(x.shape)

        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        #print(x.shape)

        return self.fc5(x)

class DQN_RAM(nn.Module):
    def __init__(self, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_RAM, self).__init__()
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=4, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(4*1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        self.rnn.flatten_parameters()
        y, _ = self.rnn(x)
        #print(y.shape)
        y = y.reshape(y.size(0), -1)
        #print(y.shape)
        x = F.relu(self.fc1(y))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

