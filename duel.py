import argparse
import time
from othelloBase import *
from canvas import Canvas
from greedyAgent import GreedyAgent
from mouseAgent import MouseAgent
from DQNAgent import DQNAgent
from MCTSAgent import MCTSAgent

# Parse command-line arguments
agents = ['greedy', 'mouse', 'DQN', 'MCTS']
parser = argparse.ArgumentParser(description='Select an agent for the Othello game.')
parser.add_argument('-b', '--black', type=str, default='greedy', choices=agents,
                    help='the black player')
parser.add_argument('-w', '--white', type=str, default='greedy', choices=agents,
                    help='the white player')
parser.add_argument('-s', '--speed', type=float, default=0, help='The speed of playing')
parser.add_argument('-v', '--visual', action='store_true', help='Whether to visualize')
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of games to play')


# Select the agent based on the command-line argument
def get_agent(agent_type, model_path=None):
    if agent_type == 'greedy':
        return GreedyAgent()
    elif agent_type == 'mouse':
        return MouseAgent()
    elif agent_type == 'DQN':
        if model_path is not None:
            return DQNAgent(model_path)
        else:
            raise ValueError("Model path must be provided for DQN agent.")
    elif agent_type == 'MCTS':
        return MCTSAgent(100)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def play_game(args):
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
    return winner


if __name__ == "__main__":
    args = parser.parse_args()
    # Create the game and canvas
    game = Othello()
    canvas = Canvas() if args.visual else None
    # initialize agents 
    agent_b = get_agent(args.black, 'RL_method/model/model_X.pth')
    agent_w = get_agent(args.white, 'RL_method/model/model_O.pth')
    # run game repeatedly
    wins = 0
    for _ in range(args.repeat):
        wins += 1 if play_game(args) else None

    win_rate = wins / args.repeat
    print(f'Win rate for black player: {win_rate * 100}%')
    if args.visual:
        canvas.draw_board(game.getBoard(), game.gamestate.getValidPositions())
        canvas.game_over(game.getWinner())