from util import *
from othelloBase import Agent, GameState
import random

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def getAction(self, state: GameState):
        possible_moves = state.getValidPositions()
        return random.choice(possible_moves)
        
        


