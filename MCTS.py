from util import *
from math import log, sqrt
import random
from othelloBase import GameState
from randomAgent import RandomAgent
from greedyAgent import GreedyAgent
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

    def UCB(self, c_param=1):
        choices_weights = [
            (c.rewards / c.visits) + c_param * ((2 * log(self.visits) / c.visits) ** 0.5)
            for c in self.children
        ]
        # random tie-breaking
        max_weight = max(choices_weights)
        best_children = [child for child, weight in zip(self.children, choices_weights) if weight == max_weight]
        return random.choice(best_children)
    
    def most_visited_child(self):
        """
        select the child visited most 
        """
        # random tie-breaking
        max_visits = max(child.visits for child in self.children)
        best_children = [child for child in self.children if child.visits == max_visits]
        return random.choice(best_children)

    def pick_random_unvisited_child(self):
        successor = self.state.generateSuccessors()
        unvisited_children = [child for child in successor if not child in self.children]
        return random.choice(unvisited_children)


class MCTS:
    def __init__(self, init_state: GameState):
        self.root = Node(init_state)
        self.visited_states = set()
        # add two agent 
        self.selfAgent = RandomAgent()
        self.opponentAgent = RandomAgent()

    def search(self, iterations):
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # select a not expanded node
            simulation_result = self.simulate(leaf)  # get the result from unfully expanded node   
            self.backpropagate(leaf, simulation_result)  # Update the information along the path
        return self.root.most_visited_child().state

    def traverse(self, node: Node):
        """
        get a node that is not visited
        """
        while node.fully_expanded() and not node.state.is_terminal():
            node = node.UCB()
        # pick up a unvisited child and add it in the tree structure 
        if not node.state.is_terminal():
            child = node.pick_random_unvisited_child()
            return node.add_child(child)
        else:
            return node 
            

    def simulate(self, leaf: Node):
        state = leaf.state
        # play until end 
        while not state.is_terminal():
            x, y = self.selfAgent.getAction(state) if self.root.state.turn == state.turn \
                  else self.opponentAgent.getAction(state)
            state = state.generateSuccessor(x, y)

        result = 0
        if state.getBlackScore() > state.getWhiteScore():
            result = 1
        elif state.getBlackScore() < state.getWhiteScore():
            result = -1
        
        # if the MCTS start with white player 
        if self.root.state.turn == -1:
            result = -result
            
        return result
    
    def backpropagate(self, node: Node, result):
        while node is not None:
            node.update(result)
            node = node.parent
