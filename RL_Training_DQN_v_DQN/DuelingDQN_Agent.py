import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import sys
import othelloEnv
from othelloBase import Othello

STATE_SPACE_LENGTH = 64
ACTION_SPACE_LENGTH = 65 # 先是64个可能的落子点；中间的四个也算上。如果没有可落子的：就是最后一个对应的。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class duelingQNet(nn.Module):
    def __init__(self):  # If in need, switch to Dueling DQN
        super(duelingQNet, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(STATE_SPACE_LENGTH, 128), nn.LeakyReLU())
        self.conv1 = nn.Conv1d(1, 4, 3, stride=1, padding=1)  
        self.conv2 = nn.Conv1d(4, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Sequential(nn.Conv1d(8, 16, 3, stride=1, padding=1), nn.LeakyReLU())
        self.linear2_val = nn.Linear(16 * 128, 512)
        self.linear2_adv = nn.Linear(16 * 128, 512)
        self.linear3_adv = nn.Linear(512, ACTION_SPACE_LENGTH)
        self.linear3_val = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = F.leaky_relu(self.conv1(x), inplace=True)  # inplace=true，输出数据会覆盖输入数据，再求梯度时不可用。
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        adv = F.leaky_relu(self.linear2_adv(x))
        adv = self.linear3_adv(adv)

        val = F.leaky_relu(self.linear2_val(x))
        val = self.linear3_val(val).expand(x.size(0), ACTION_SPACE_LENGTH)
        out = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), ACTION_SPACE_LENGTH)
        return out


class DQNAgent():
    def __init__(self, game: Othello, turn):
        self.q_net = duelingQNet().to(device)
        if turn == 1:  # X
            self.q_net.load_state_dict(torch.load('model/model_offensive.pth'))
        else:  # O
            self.q_net.load_state_dict(torch.load('model/model_defensive.pth'))
        self.q_net.eval()
        self.game = game
        self.turn = turn

    def updateGame(self, game: Othello):
        self.game = game

    def findBestMove(self):
        valid_pos = self.game.getValidPositions(self.turn)
        # if len(valid_pos)==0:
        #     return 64  # 实际上是第65个，表示当前玩家没有可行位置，本轮被跳过
        x = torch.tensor(self.game.board, dtype=torch.float32).flatten().to(device)
        action_values = self.q_net(x)
        valid_indices = [8 * row + col for (row, col) in valid_pos]
        valid_q_values = action_values[valid_indices]
        action = valid_pos[torch.argmax(valid_q_values)]
        return action
    
    def makeMove(self):
        '''
        Play the best move directly.
        '''
        best_move = self.findBestMove()
        # print(best_move)
        if best_move is not None:
            x, y = best_move
            self.game.place(x, y, self.turn)