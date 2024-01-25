from util import *
from MCTS import MCTS
from othelloBase import Agent, GameState
from greedyAgent import GreedyAgent
from randomAgent import RandomAgent
from DQNAgent import DQNAgent
from minmaxAgent import MinmaxAgent
import numpy as np

class MCTSAgent(Agent):
    def __init__(self, index, iterations=20, default_agent="random"):
        super().__init__(index)
        self.iterations = iterations

        if default_agent == 'random':
            self.default_policy = RandomAgent
        elif default_agent == 'greedy':
            self.default_policy = GreedyAgent
        elif default_agent == 'DQN':
            self.default_policy = DQNAgent
            
    def getAction(self, state: GameState):  
        mcts = MCTS(state, self.default_policy)
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
        base = self.iterations
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


