'''
Based on the othelloBase, 
implement a greedy agent that plays the Othello game,
which, for every turn, chooses the move that flips the most pieces.
'''
from othelloBase import Othello

class greedyAgent():
    def __init__(self, game: Othello):
        self.game = game
        self.max_flips = 0
        self.best_move = None
    
    def updateGame(self, game: Othello):
        self.game = game

    def findBestMove(self):
        '''
        Find the best move.
        '''
        self.best_move = None
        self.max_flips = 0
        self.turn = self.game.getTurn()
        # print(self.turn)
        self.valid_positions = self.game.getValidPositions(self.turn)
        # print(f"Valid positions: {self.valid_positions}")
        for (x, y) in self.valid_positions:
            flips = self.game.place(x, y, self.turn)
            if flips > self.max_flips:
                self.max_flips = flips
                self.best_move = (x, y)
            self.game.undo()
        
        return self.best_move
    
    def makeMove(self):
        '''
        Play the best move directly.
        '''
        best_move = self.findBestMove()
        # print(best_move)
        if best_move is not None:
            x, y = best_move
            self.game.place(x, y, self.turn)
