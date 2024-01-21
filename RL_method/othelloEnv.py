'''
Othello, but interfaces of gym enviroment implemented.

感觉没用
'''

import gymnasium as gym
from copy import deepcopy
import sys
sys.path.append("..")
from othelloBase import Othello

class OthelloEnvironment(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.game = Othello() # game == state

    def step(self, action:int):
        # actions: 0 - 64
        # 64: pass (no valid moves)
        # else: 8*row + col
        row = action // 8
        col = action % 8
        reward = 0
        # by default, we believe the actions are valid
        if action == 64:
            pass
        else:
            self.game.place(row, col, self.game.turn)
        
        if self.game.isEnd():
            winner = self.game.getWinner()
            score = self.game.getScore()
            # Print the winner and score
            # if winner == 1:
            #     print("Black wins!")
            # elif winner == -1:
            #     print("White wins!")
            # else:
            #     print("It's a tie!")
            # print(f"Score for X: {score}")
            reward = score * 10
            terminated = True
            return self.game, reward, terminated, False, {}
        else:
            # 对在边角的情况进行奖励
            if (row,col) == (0,0) or (row,col) == (0,7) or (row,col) == (7,0) or (row,col) == (7,7):
                reward += 100
            elif row == 0 or row == 7 or col == 0 or col == 7:
                reward += 20
            # 对于下一步会导致对方下到角落的情况进行惩罚
            else:
                l = self.game.getValidPositions(-self.game.turn)
                if (0,0) in l or (0,7) in l or (7,0) in l or (7,7) in l:
                    reward -= 50
                else:
                    reward = 0
            terminated = False
            return self.game, reward, terminated, False, {}

    def reset(self):
        self.game.board = [[0 for _ in range(8)] for _ in range(8)]
        self.game.board[3][3] = self.game.board[4][4] = -1
        self.game.board[3][4] = self.game.board[4][3] = 1
        self.game.turn = 1
        self.game._boardHistory = [deepcopy(self.game.board)]
        self.game._turnHistory = [self.game.turn]
        self.game._flipCount = 0
        self.game._checkCount = 0

        return self.game

    def render(self, mode = "human"):
        pass

    def close(self):
        pass