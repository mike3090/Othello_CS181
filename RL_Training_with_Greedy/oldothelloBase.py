'''
Implementation of a simple but hard game: Othello.

https://zh.wikipedia.org/wiki/%E9%BB%91%E7%99%BD%E6%A3%8B
'''

from copy import deepcopy
import numpy as np

class Othello():
    # Constants
    SIZE = 8
    def __init__(self):
        '''
        A 8*8 board, 1 for X / black, -1 for O/ white, 0 for empty.

        Initialize order:
          0 1 2 3 4 5 6 7
        0 . . . . . . . .
        1 . . . . . . . .
        2 . . . . . . . .
        3 . . . O X . . .
        4 . . . X O . . .
        5 . . . . . . . .
        6 . . . . . . . .
        7 . . . . . . . .

        Naming standard: the upper right X is [3][4]. 先行，后列。

        X / Black goes first.
        '''
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
        self.turn = 1
        self._boardHistory = [deepcopy(self.board)]
        self._turnHistory = [self.turn]
        self._flipCount = 0
        self._checkCount = 0

    def printBoard(self):
        '''
        Return a string representation of the board.
        '''
        print(
            '\n'.join(["  0 1 2 3 4 5 6 7"]+[str(i)+" "+(' '.join(map(lambda x: 'X' if x == 1 else 'O' if x == -1 else '.', self.board[i]))) for i in range(len(self.board))])
        )
    
    def getValidPositions(self, turn):
        '''
        Return a list of valid positions / moves for the current player.
        '''
        
        validPositions = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 0 and self._checkValid(i, j, turn):
                    validPositions.append((i, j))
        return validPositions
    
    def _checkValid(self, x, y, turn):
        '''
        Check if a move is valid for the current player.
        '''
        if self.board[x][y] != 0:
            return False
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                self._checkCount = 0
                if dx == dy == 0: # therefore it covers all 8 directions
                    continue
                if self._checkDirection(x, y, dx, dy, turn):
                    return True
        return False
    
    def _checkDirection(self, x, y, dx, dy, turn):
        '''
        Check if a move is valid for the current player in a specific direction.
        '''
        if x + dx < 0 or x + dx >= 8 or y + dy < 0 or y + dy >= 8:
            return False
        if self.board[x + dx][y + dy] == -turn:
            self._checkCount += 1
            return self._checkDirection(x + dx, y + dy, dx, dy, turn)
        elif self.board[x + dx][y + dy] == turn:
            if self._checkCount != 0:
                return True
        else:
            return False
    
    def place(self, x, y, turn):
        '''
        Place a piece on the board.
        '''
        self.board[x][y] = turn
        flips = self._flip(x, y, turn)
        self.turn = -turn
        # self._updateCount()
        validPositions = self.getValidPositions(self.turn)
        if len(validPositions) == 0:
            self.turn = -self.turn # 变回来，因为这个时候下一个玩家没有合法的位置，所以当前玩家继续落子
            validPositions = self.getValidPositions(self.turn)
        self._boardHistory.append(deepcopy(self.board))
        self._turnHistory.append(self.turn)

        return flips
    
    def _flip(self, x, y, turn):
        '''
        Flip pieces on the board.
        '''
        self._flipCount = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0:
                    continue
                if self._checkDirection(x, y, dx, dy, turn):
                    self._flipDirection(x, y, dx, dy, turn)
        return self._flipCount

    def _flipDirection(self, x, y, dx, dy, turn):
        '''
        Flip pieces on the board in a specific direction.
        '''
        if x + dx < 0 or x + dx >= 8 or y + dy < 0 or y + dy >= 8:
            return
        if self.board[x + dx][y + dy] == -turn:
            self.board[x + dx][y + dy] = turn
            self._flipCount += 1
            self._flipDirection(x + dx, y + dy, dx, dy, turn)
        elif self.board[x + dx][y + dy] == turn:
            return
        else:
            self.board[x + dx][y + dy] = turn

    def undo(self):
        '''
        Undo the move.
        '''
        self._flipCount = 0
        self._boardHistory.pop()
        self._turnHistory.pop()
        
        self.board = deepcopy(self._boardHistory[-1])
        self.turn = self._turnHistory[-1]
        

    def isEnd(self):
        '''
        Check if the game is over.
        '''
        if self.getValidPositions(1) == [] and self.getValidPositions(-1) == []:
            return True
        else:
            return False
        
    def getWinner(self):
        '''
        Return the winner.
        '''
        blackCount = 0
        whiteCount = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1:
                    blackCount += 1
                elif self.board[i][j] == -1:
                    whiteCount += 1
        
        if whiteCount > blackCount:
            return -1
        elif whiteCount < blackCount:
            return 1
        else:
            return 0
    
    def getScore(self):
        '''
        Return the score, 
        for the winner, not X,
        so it's always non-negative.
        '''
        blackCount = 0
        whiteCount = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1:
                    blackCount += 1
                elif self.board[i][j] == -1:
                    whiteCount += 1
        
        return abs(blackCount - whiteCount)
    
    def getBoard(self):
        '''
        Return the board.
        '''
        return self.board
    
    def getTurn(self):
        '''
        Return the current player.
        '''
        return self.turn
    
    def getState(self):
        """返回当前棋盘的状态

        Returns:
            64维向量
        """
        return np.array(self.board, dtype=np.int).flatten()     # 返回一个一维数组
    
    def Add(self, my_color, pos):
        """加入一个新的棋子

        Arguments:
            my_color {int} -- 1表示黑棋，-1表示白棋
            pos {int} -- 位置
        """
        if pos != 64:
            (x, y) = (pos // self.board_size, pos % self.board_size)
            self.board[x][y] = my_color
            if my_color == 1:
                self.black_chess.add((x, y))
                self.board[x][y] = 1
            elif my_color == -1:
                self.white_chess.add((x, y))
                self.board[x][y] = -1

            if my_color == 1:
                self.Reverse((x, y), self.black_chess, self.white_chess, my_color)
            elif my_color == -1:
                self.Reverse((x, y), self.white_chess, self.black_chess, my_color)
