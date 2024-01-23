from util import *
from MCTS import MCTS
from othelloBase import Agent, GameState

class MCTSAgent(Agent):
    def __init__(self, iterations=100):
        super().__init__()
        self.iterations = iterations

    def getAction(self, state: GameState):  
        mcts = MCTS(state)
        actions = state.getValidPositions()
        # to adjust iterations by the borad stones 
        factor = (state.getBlackScore() + state.getWhiteScore()) / 64 + 0.5
        iterations = int(self.iterations * factor)

        next_state = mcts.search(iterations)
        for x, y in actions:
            if next_state == state.generateSuccessor(x, y):
                return x, y
        return None
        
        


