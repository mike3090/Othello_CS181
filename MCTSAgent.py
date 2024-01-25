from util import *
from MCTS import MCTS
from othelloBase import Agent, GameState
import numpy as np

class MCTSAgent(Agent):
    def __init__(self, index, iterations=20):
        super().__init__(index)
        self.iterations = iterations

    def getAction(self, state: GameState):  
        mcts = MCTS(state)
        actions = state.getValidPositions()

        # to adjust iterations by the borad stones 
        next_state = mcts.search(self.adjust_iterations(state))
        for x, y in actions:
            if next_state == state.generateSuccessor(x, y):
                return x, y
        return None
        

    def adjust_iterations(self, state):
        """
        Gaussian function to adjust iterations
        """
        base = 20
        total_score = state.getBlackScore() + state.getWhiteScore()
        x = total_score / 64  # Normalize score to [0, 1]
        # Gaussian function parameters
        mu = 0.5  # Mean (peak at the middle of the game)
        sigma = 0.1  # Standard deviation (controls the width of the bell curve)
        # Calculate factor using Gaussian function
        factor = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        # Adjust iterations
        iterations = int(self.iterations * factor) + base 
        return iterations       


