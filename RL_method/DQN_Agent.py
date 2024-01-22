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

class QNet(nn.Module):
    ''' Q网络 '''
    def __init__(self): #switch to Dueling DQN
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(STATE_SPACE_LENGTH, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4A = nn.Linear(128, ACTION_SPACE_LENGTH)
        self.fc4V = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        Aout = self.fc4A(x)
        Vout = self.fc4V(x)
        out = Vout + Aout - Aout.mean()
        return out

class DQNAgent():
    def __init__(self, game: Othello, turn):
        self.q_net = QNet()
        if turn == 1: # X
            self.q_net.load_state_dict(torch.load('model/model_X.pth'))
        else: # O
            self.q_net.load_state_dict(torch.load('model/model_O.pth'))
        self.q_net.to(device)
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