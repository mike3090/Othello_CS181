from util import *
from othelloBase import GameState
from othelloBase import Agent
import random

def scoreEvaluationFunction(currentGameState: GameState, color):
		"""
		This evaluation function is composed of...
		simple evaluation function, slightly smarter than greedy
		"""
		w = currentGameState.getWhiteScore()
		b = currentGameState.getBlackScore()
		if color == 'black':
			return b - w
		elif color == 'white':
			return w - b


class MinmaxAgent(Agent):
	def __init__(self, color):
		# depth of the search tree
		self.depth = 2
		self.index = 0
		self.color = color
	
	def getAction(self, state: GameState):
			def minmax(gamestate: GameState, depth, index):
						nextDepth = (depth - 1) if index == 0 else depth
						if nextDepth == -1 or gamestate.getBlackScore() + gamestate.getWhiteScore() == 64:
								return scoreEvaluationFunction(gamestate, self.color), None
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
								value, childaction = minmax(newstate, nextDepth, nextindex)
								if value == f(value, v):
										v = value
										a = move
						return v, a
			_, A = minmax(state, self.depth, self.index)
			return A