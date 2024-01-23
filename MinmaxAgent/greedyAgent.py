'''
Based on the othelloBase, 
implement a greedy agent that plays the Othello game,
which, for every turn, chooses the move that flips the most pieces.
'''
from util import *
from othelloBase import GameState
from othelloBase import Agent
import random

class GreedyAgent(Agent):

    def getAction(self, state: GameState):
        '''
        Find the best move.
        '''
        best_move = None
        valid_positions = state.getValidPositions()
        max_flips = 0
        #print(f"Valid positions: {valid_positions}")
        for (x, y) in valid_positions:
            flips = state.countFlips(x, y)
            if flips > max_flips:
                max_flips = flips
                best_move = (x, y)
            # introduce some randomness.
            if flips == max_flips:
                if random.random() > 0.5:
                    max_flips = flips
                    best_move = (x, y)
        if best_move is None:
          print("No valid move found. Exiting the program.")  
        return best_move