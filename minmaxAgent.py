from util import *
from othelloBase import GameState
from othelloBase import Agent
import random

def scoreEvaluationFunction(currentGameState: GameState, color):
		"""
		This evaluation function is composed of:
		1. how many pieces does the chosen color own more than its opponent
		2. how many edge pieces does the chosen color own
		3. how many corner pieces does the chosen color own
		"""

		# feature 1
		w = currentGameState.getWhiteScore()
		b = currentGameState.getBlackScore()
		if color == 'black':
			score_1 = b - w
			index = 1
		elif color == 'white':
			score_1 = w - b
			index = -1
		
		# feature 2
		safe_points = set()

		for point in [(0, 0), (0, 7), (7, 0), (7, 7)]:
			if currentGameState.board[point[0]][point[1]] == index:
				safe_points.add(point)
		corner_num = len(safe_points)

		def find_connected_points(currentGameState, point, index, direction): # recursive search function, search for connected parts of corner points
			connected_points = []
			next_point = (point[0] + direction[0], point[1] + direction[1])
			if 0 <= next_point[0] < 8 and 0 <= next_point[1] < 8:
				if currentGameState.board[next_point[0]][next_point[1]] == index:
					connected_points.append(next_point)
					connected_points += find_connected_points(currentGameState, next_point, index, direction)
			return connected_points
		
		if (0,0) in safe_points:
			safe_points.update(find_connected_points(currentGameState, (0, 0), index, (1, 0)))
			safe_points.update(find_connected_points(currentGameState, (0, 0), index, (0, -1)))

		if (0,7) in safe_points:
			safe_points.update(find_connected_points(currentGameState, (0, 7), index, (-1, 0)))
			safe_points.update(find_connected_points(currentGameState, (0, 7), index, (0, -1)))

		if (7,0) in safe_points:
			safe_points.update(find_connected_points(currentGameState, (7, 0), index, (1, 0)))
			safe_points.update(find_connected_points(currentGameState, (7, 0), index, (0, 1)))

		if (7,7) in safe_points:
			safe_points.update(find_connected_points(currentGameState, (7, 7), index, (-1, 0)))
			safe_points.update(find_connected_points(currentGameState, (7, 7), index, (0, 1)))

		score_2 = len(safe_points) - corner_num

		# feature 3
		score_3 = 0
		for point in [(0, 0), (0, 7), (7, 0), (7, 7)]:
			if currentGameState.board[point[0]][point[1]] == index:
				score_3 += 1
			if currentGameState.board[point[0]][point[1]] == -index:
				score_3 -= 1

		# weights for each score
		w1 = 1.0
		w2 = 1.0
		w3 = 5.0
		
		return w1* score_1 + w2 * score_2 + w3* score_3


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