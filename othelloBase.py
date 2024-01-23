'''
Implementation of a simple but hard game: Othello.

https://zh.wikipedia.org/wiki/%E9%BB%91%E7%99%BD%E6%A3%8B
'''

from util import *
from copy import deepcopy
import numpy as np

class Othello():
    # Constants
    SIZE = 8
    def __init__(self):
        self.gamestate = GameState()
        self._boardHistory = [deepcopy(self.gamestate.board)]
        self._turnHistory = [self.gamestate.turn]

    def printBoard(self):
        '''
        Return a string representation of the board.
        '''
        print(
            '\n'.join(["  0 1 2 3 4 5 6 7"]+[str(i)+" "+(' '.join(map(lambda x: 'X' if x == 1 else 'O' if x == -1 else '.', self.gamestate.board[i]))) for i in range(len(self.gamestate.board))])
        )
    
    def place(self, x, y):
        '''
        Place a piece on the board.
        '''
        self.gamestate = self.gamestate.generateSuccessor(x,y)
        self._boardHistory.append(deepcopy(self.gamestate.board))
        self._turnHistory.append(self.gamestate.turn)
    
    def undo(self):
        '''
        Undo the move.
        '''
        self._boardHistory.pop()
        self._turnHistory.pop()
        
        self.gamestate.board = deepcopy(self._boardHistory[-1])
        self.gamestate.turn = self._turnHistory[-1]
        

    def isEnd(self):
        '''
        Check if the game is over.
        '''
        return self.gamestate.is_terminal()
        
    def getWinner(self):
        '''
        Return the winner.
        '''
        blackCount = self.gamestate.getBlackScore()
        whiteCount = self.gamestate.getWhiteScore()

        if whiteCount > blackCount:
            return -1
        elif whiteCount < blackCount:
            return 1
        else:
            return 0
    
    def getScore(self):
        '''
        Return the score, 
        currently for black X.
        '''
        return self.gamestate.getBlackScore() - self.gamestate.getWhiteScore()
    
    def getBoard(self):
        '''
        Return the board.
        '''
        return self.gamestate.board
    
    def getTurn(self):
        '''
        Return the current player.
        '''
        return self.gamestate.turn

class GameState:
    def __init__(self, board = None, turn = None):
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
        self.board = board
        self.turn = turn
        if board is None:
            self.board = [[0 for _ in range(8)] for _ in range(8)]
            self.board[3][3] = self.board[4][4] = -1
            self.board[3][4] = self.board[4][3] = 1
        if turn is None:
            self.turn = 1

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self.board == other.board and self.turn == other.turn
        return False  
    
    def reverseTurn(self):
        self.turn *= -1

    def getValidPositions(self):
        '''
        Return a list of valid positions / moves for the current player.
        '''
        validPositions = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 0 and self._checkValid(i, j):
                    validPositions.append((i, j))
        return validPositions

    def _checkValid(self, x, y):
        '''
        Check if a move is valid for the current player.
        '''
        if self.board[x][y] != 0:
            return False
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                if dx == dy == 0: # therefore it covers all 8 directions
                    continue
                if self._isValidDirection(x, y, dx, dy):
                    return True
        return False

    def _isValidDirection(self, x, y, dx, dy):
        '''
        Check if a move is valid for the current player in a specific direction.
        '''
        i, j = x + dx, y + dy
        if i < 0 or i >= 8 or j < 0 or j >= 8 or self.board[i][j] != -self.turn:
            return False
        while 0 <= i < 8 and 0 <= j < 8:
            if self.board[i][j] == self.turn:
                return True
            elif self.board[i][j] == -self.turn:
                i, j = i + dx, j + dy
            else:
                 return False
        return False
    
    def generateSuccessors(self):
        '''
        Return a list of valid successors.
        '''
        successors = []
        validPositions =self.getValidPositions()
        for x, y in validPositions:
            successors.append(self.generateSuccessor(x, y))
        return successors

    def generateSuccessor(self, x, y):
        '''
        return a gamestate after placing stone in (x,y)
        '''
        successor = deepcopy(self)
        successor.board[x][y] = self.turn
        successor._flip(x, y)
        successor.reverseTurn()
        if successor.getValidPositions() == []:
            successor.reverseTurn()

        return successor
    
    def countFlips(self, x, y):
        '''
        Return the number of stones that would be flipped if a stone were placed at (x, y).
        '''
        flipCount = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0:
                    continue
                i, j = x + dx, y + dy
                while 0 <= i < 8 and 0 <= j < 8 and self.board[i][j] == -self.turn:
                    flipCount += 1
                    i, j = i + dx, j + dy
        return flipCount

    def _flip(self, x, y):
        '''
       Flip pieces on the board if put stone on (x,y).
        '''
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0:
                    continue
                if self._isValidDirection(x, y, dx, dy):
                    i, j = x + dx, y + dy
                    while 0 <= i < 8 and 0 <= j < 8 and self.board[i][j] == -self.turn:
                        self.board[i][j] = self.turn
                        i, j = i + dx, j + dy


    def getBlackScore(self):
        BlackCount = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1:
                    BlackCount += 1
        return BlackCount

    def getWhiteScore(self):
        whiteCount = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == -1:
                    whiteCount += 1
        return whiteCount
    
    def is_terminal(self):
        return self.getValidPositions() == []

class Agent:
    """
    An agent must define a getAction method
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state: GameState):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()