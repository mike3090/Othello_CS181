import argparse
import time
from othelloBase import *
from canvas import Canvas
from greedyAgent import GreedyAgent
from mouseAgent import MouseAgent
from DQNAgent import DQNAgent

# Parse command-line arguments
agents = ['greedy', 'mouse', 'DQN']
parser = argparse.ArgumentParser(description='Select an agent for the Othello game.')
parser.add_argument('-b', '--black', type=str, default='greedy', choices=agents,
                    help='the black player')
parser.add_argument('-w', '--white', type=str, default='greedy', choices=agents,
                    help='the white player')
parser.add_argument('-s', '--speed', type=float, default=0, help='The speed of playing')
parser.add_argument('-v', '--visual', action='store_true', help='Whether to visualize')

args = parser.parse_args()

# Create the game and canvas
game = Othello()
canvas = Canvas() if args.visual else None

# Select the agent based on the command-line argument
if args.black == 'greedy':
    agent_b = GreedyAgent()
elif args.black == 'mouse':
    agent_b = MouseAgent()
elif args.black == 'DQN':
    agent_b = DQNAgent('RL_method/model/model_X.pth')



if args.white == 'greedy':
    agent_w = GreedyAgent()
elif args.white == 'mouse':
    agent_w = MouseAgent()
elif args.white == 'DQN':
    agent_w = DQNAgent('RL_method/model/model_O.pth')

# Main game loop
while not game.isEnd():
    state = game.gamestate

    canvas.draw_board(state.board, state.getValidPositions()) if args.visual else None

    # control the playing speed
    time.sleep(args.speed)
    if state.turn == 1:
        x, y = agent_b.getAction(state)
    else:
        x, y = agent_w.getAction(state)

    game.place(x, y)

winner = "Black" if game.getWinner() == 1 else "White" if game.getWinner() == -1 else "No one (it's a tie)"
print(f"Game Over. {winner} wins!")

if args.visual:
    canvas.draw_board(game.getBoard(), game.gamestate.getValidPositions())
    canvas.game_over(game.getWinner())