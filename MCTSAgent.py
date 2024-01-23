from util import *
from MCTS import MCTS
from othelloBase import Agent, GameState

class MCTSAgent(Agent):
    def __init__(self, iterations=1000):
        super().__init__()
        self.iterations = iterations

    def getAction(self, state: GameState):  
        mcts = MCTS(state)
        actions = state.getValidPositions()
        next_state = mcts.search(self.iterations)
        for x, y in actions:
            if next_state == state.generateSuccessor(x, y):
                return x, y
        return None
        
        


