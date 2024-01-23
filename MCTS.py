from util import *
from math import log, sqrt
import random
from othelloBase import GameState
import pickle

class Node:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def add_child(self, state:GameState):
        """
        Add the node into treee and return the child node 
        """
        child = Node(state, self)
        self.children.append(child)
        return child 
        
    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def fully_expanded(self):
        return len(self.children) == len(self.state.generateSuccessors())

    def best_child(self, c_param=1.44):
        choices_weights = [
            (c.rewards / c.visits) + c_param * ((2 * log(self.visits) / c.visits) ** 0.5)
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def pick_random_unvisited_child(self):
        successor = self.state.generateSuccessors()
        unvisited_children = [child for child in successor if not child in self.children]
        return random.choice(unvisited_children)


class MCTS:
    def __init__(self, init_state: GameState):
        self.root = Node(init_state)

    def search(self, iterations):
        # print("*************root**************")
        # printBoard(self.root.state.board)
        # print("*************root**************")
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # select a not expanded node
            if leaf is None:
                # reach the terminate state 
                continue  
            simulation_result = self.simulate(leaf)  # get the result from unfully expanded node   
            self.backpropagate(leaf, simulation_result)  # Update the information along the path

        # print("*************root exit**************")
        # printBoard(self.root.state.board)
        # print(self.root.state.is_terminal())
        # print("*************root exit**************")
        return self.root.best_child(c_param=0.).state

    def traverse(self, node: Node):
        """
        get a node that is not visited
        """
        while node.fully_expanded():
            # if the node is the terminate state return 
            if node.state.is_terminal():
                return None
            node = node.best_child()
        # pick up a unvisited child and add it in the tree structure 
        child = node.pick_random_unvisited_child()
        return node.add_child(child)

    def simulate(self, leaf: Node):
        state = leaf.state
        # play until end 
        while not state.is_terminal():
            # printBoard(state.board)
            # print(state.turn)
            # print ("******************")
            x, y = self.choose_random_action(state)
            state = state.generateSuccessor(x, y)

        result = 0
        if state.getBlackScore() > state.getWhiteScore():
            result = 1
        elif state.getBlackScore() < state.getWhiteScore():
            result = -1
        return result
    
    def backpropagate(self, node: Node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def choose_random_action(self, state: GameState):
        """
        random agent 
        """
        possible_moves = state.getValidPositions()
        return random.choice(possible_moves)