import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import sys
import othelloEnv
# from othelloEnv import OthelloEnvironment
sys.path.append("..")
from othelloBase import Othello


STATE_SPACE_LENGTH = 64
ACTION_SPACE_LENGTH = 65 # 先是64个可能的落子点；中间的四个也算上。如果没有可落子的：就是最后一个对应的。

LR = 0.001
EPISODE = 10000
BATCH_SIZE = 60
GAMMA = 0.9
ALPHA = 0.8
BUFFER_CAPACITY = 10000
MINIMAL_BUFFER_SIZE = 100
UPDATE_FREQUENCY = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:  # Thanks to: Hands on RL
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, terminated):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class duelingQNet(nn.Module):
    ''' duelingQ网络 '''
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


class DuelingDQN(object):
    def __init__(self, turn):
        self.turn = turn
        self.duelingQNet, self.target_q_Net = duelingQNet().to(device), duelingQNet().to(device)
        self.optimizer = torch.optim.Adam(self.duelingQNet.parameters(),lr=LR)
        self.epsilon = 0.8
        self.count = 0

    def choose_action(self, state: Othello):
        self.epsilon = max(0.01, 0.8-self.count*0.001)
        valid_pos = state.getValidPositions(self.turn)
        if len(valid_pos) == 0:
            return 64
        if random.random() < self.epsilon:
            action = random.choice(valid_pos)
        else:
            x = torch.tensor(state.board, dtype=torch.float32).flatten().to(device)
            action_values = self.duelingQNet(x)
            valid_indices = [8 * row + col for (row, col) in valid_pos]
            valid_q_values = action_values[valid_indices]
            action = valid_pos[torch.argmax(valid_q_values)]
        # convert action (row, col) to a single int = 8*row+col (align with 64)
        row, col = action
        action = 8*row+col
        return action

    def update(self, transition_dict, oppo_dueling_q_net):  # same as DQN
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int32).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions).to(device)
        # next_q_values = self.target_q_net(next_states)
        # max_next_q_values = next_q_values.max(1)[0].view(-1,1)  # max(1) return (value,index) for each row
        # q_targets = rewards + GAMMA * max_next_q_values * (1 - dones).to(device)
        q_values_oppo = oppo_dueling_q_net(next_states).detach()
        max_q_values_oppo = torch.max(q_values_oppo, 1)[0].view(-1, 1)  # max(1) return (value,index) for each row
        q_targets = rewards + GAMMA * max_q_values_oppo * (1 - dones).to(device)
        '''
        - Gamma? if encounter problem, check
        '''

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % UPDATE_FREQUENCY == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    env = othelloEnv.OthelloEnvironment()
    playerX = DuelingDQN(1)   # offensive
    playerO = DuelingDQN(-1)  # defensive
    bufferX = ReplayBuffer(BUFFER_CAPACITY)
    bufferO = ReplayBuffer(BUFFER_CAPACITY)
    return_list_X = []
    return_list_O = []

    for episode in range(EPISODE):
        # print(f"----------EPISODE{episode}------------")
        episode_return_X = 0
        episode_return_O = 0
        game_state = env.reset()
        terminated = False
        while True:
            # 先手 X
            # game_state.printBoard()
            state = env.game
            action = playerX.choose_action(state)
            next_state, reward, terminated, _, _ = env.step(action)
            bufferX.add(state.board, action, reward, next_state.board, terminated)
            state = next_state
            episode_return_X += reward
            episode_return_O -= reward # 因为reward是从X的角度来看的，所以O的reward要取反，就直接-=了

            if terminated:
                if bufferX.size() > MINIMAL_BUFFER_SIZE:
                    b_s, b_a, b_r, b_ns, b_d = bufferX.sample(BATCH_SIZE)
                    b_s, b_ns = np.reshape(b_s, (BATCH_SIZE, 64)), np.reshape(b_ns, (BATCH_SIZE, 64))
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    playerX.update(transition_dict, playerO.target_q_net)
                break
            # 后手 O
            # game_state.printBoard()
            state = env.game
            action = playerO.choose_action(state)
            next_state, reward, terminated, _, _ = env.step(action)
            bufferO.add(state.board, action, reward, next_state.board, terminated)
            state = next_state
            episode_return_X += reward
            episode_return_O -= reward

            if terminated:
                if bufferO.size() > MINIMAL_BUFFER_SIZE:
                    b_s, b_a, b_r, b_ns, b_d = bufferO.sample(BATCH_SIZE)
                    b_s, b_ns = np.reshape(b_s, (BATCH_SIZE, 64)), np.reshape(b_ns, (BATCH_SIZE, 64))
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    playerO.update(transition_dict, playerO.target_q_net)
                break
        return_list_X.append(episode_return_X)
        return_list_O.append(episode_return_O)
        # print(f"X: {episode_return_X}, O: {episode_return_O}")
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: X: {episode_return_X}, O: {episode_return_O}")
            torch.save(playerX.q_net.state_dict(), "data/model_offensive.pth")
            torch.save(playerO.q_net.state_dict(), "data/model_defensive.pth")
