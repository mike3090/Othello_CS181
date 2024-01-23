import sys
sys.path.append("./RL_method")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from othelloBase import Agent, GameState

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

class DQNAgent(Agent):
    def __init__(self, ckpt_path: str):
        self.q_net = QNet().to(device)
        self.q_net.load_state_dict(torch.load(ckpt_path))
        self.q_net.eval()

    def getAction(self, state: GameState):
        valid_pos = state.getValidPositions()
        # if len(valid_pos)==0:
        #     return 64  # 实际上是第65个，表示当前玩家没有可行位置，本轮被跳过
        x = torch.tensor(state.board, dtype=torch.float32).flatten().to(device)
        action_values = self.q_net(x)
        valid_indices = [8 * row + col for (row, col) in valid_pos]
        valid_q_values = action_values[valid_indices]
        action = valid_pos[torch.argmax(valid_q_values)]
        return action