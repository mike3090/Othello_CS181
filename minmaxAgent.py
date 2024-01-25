from util import *
from othelloBase import GameState
from othelloBase import Agent
import random

class MinmaxAgent(Agent):
	def __init__(self, index, pruning = True):
		super().__init__(index)
		# depth of the search tree
		self.depth = 2
		self.pruning = pruning
		
	def minmax(self, gamestate: GameState, depth, index, alpha, beta):
		nextDepth = (depth - 1) if index == 0 else depth
		if nextDepth == -1 or gamestate.getBlackScore() + gamestate.getWhiteScore() == 64:
				return scoreEvaluationFunction(gamestate, self.index), None
		if index == 0:
				v = -999
				a = None
				f = max
		else:
				v = 999 
				a = None
				f = min
		nextindex = index + 1
		nextindex = nextindex % 2 # there are 2 agents, minmaxagent and opponent

		for move in gamestate.getValidPositions():
			newstate = gamestate.getNextState(move)
			value, childaction = self.minmax(newstate, nextDepth, nextindex, alpha, beta)
			if value == f(value, v):
					v = value
					a = move
			if self.pruning:
				if index == 0:
					if value > beta:
							return v,a
					alpha = max(alpha, value)
				else:
					if value < alpha:
							return v,a
					beta = min(beta, value)
		return v, a
	
	def getAction(self, state: GameState):
		_, A = self.minmax(state, self.depth, self.index, -999, 999)
		return A




		