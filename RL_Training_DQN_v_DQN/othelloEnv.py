'''
Othello, but interfaces of gym enviroment implemented.

感觉没用
'''

import gymnasium as gym
from copy import deepcopy
from oldothelloBase import Othello

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
            elif (row, col) == (0,1) or (row, col) == (1,0) or (row, col) == (1,1) \
                 or (row, col) ==(0,6) or (row, col) ==(1,6) or (row, col) ==(1,7) \
                 or (row, col) ==(6,0) or (row, col) ==(6,1) or (row, col) ==(7,1) \
                 or (row, col) ==(6,6) or (row, col) ==(6,7) or (row, col) ==(7,6):
                reward -=35
            elif (row, col) == (0,2) or (row, col) == (0,5) or (row, col) == (2,0) \
                 or (row, col) == (2,7) or (row, col) == (6,0) or (row, col) == (6,7) \
                 or (row, col) == (7,2) or (row, col) == (7,5):
                reward += 10
            elif (row, col) == (3, 0) or (row, col) == (4, 0) \
                 or (row, col) == (2, 2) or (row, col) == (5, 2) \
                 or (row, col) == (2, 5) or (row, col) == (5, 5) \
                 or (row, col) == (3, 7) or (row, col) == (4, 7):
                reward += 5
            elif (row, col) == (2, 1) or (row, col) == (3, 1) or (row, col) == (4, 1) or (row, col) == (5, 1) \
                 or (row, col) == (1, 2) or (row, col) == (6, 2) or (row, col) == (1, 3) or (row, col) == (6, 3) \
                 or (row, col) == (1, 4) or (row, col) == (6, 4) or (row, col) == (1, 5) or (row, col) == (6, 5) \
                 or (row, col) == (3, 4) or (row, col) == (4, 4) or (row, col) == (3, 3) or (row, col) == (4, 3) \
                 or (row, col) == (2, 6) or (row, col) == (3, 6) or (row, col) == (4, 6) or (row, col) == (5, 6):
                reward += 2
            elif (row, col) == (3, 2) or (row, col) == (4, 2) or (row, col) == (2, 3) or (row, col) == (5, 3) \
                 or (row, col) == (2, 4) or (row, col) == (5, 4) or (row, col) == (3, 5) or (row, col) == (4, 5):
                reward += 1
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