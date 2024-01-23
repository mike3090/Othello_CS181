import argparse
import time
from othelloBase import *
from canvas import Canvas
from greedyAgent import GreedyAgent
from mouseAgent import MouseAgent
from DQNAgent import DQNAgent
from MCTSAgent import MCTSAgent
from minmaxAgent import MinmaxAgent
from minmaxAgent import AlphaBetaAgent
import random

# Parse command-line arguments
agents = ['greedy', 'mouse', 'DQN', 'MCTS', 'minmax', 'alphabeta']
parser = argparse.ArgumentParser(description='Select an agent for the Othello game.')
parser.add_argument('-b', '--black', type=str, default='greedy', choices=agents,
                    help='the black player')
parser.add_argument('-w', '--white', type=str, default='greedy', choices=agents,
                    help='the white player')
parser.add_argument('-s', '--speed', type=float, default=0, help='The speed of playing')
parser.add_argument('-v', '--visual', action='store_true', help='Whether to visualize')
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of games to play')


# Select the agent based on the command-line argument
def get_agent(agent_type, model_path=None, color = None, weights = None):
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
    elif agent_type == 'minmax':
        return MinmaxAgent(color, weights)
    elif agent_type == 'alphabeta':
        return AlphaBetaAgent(color, weights)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def play_game(args, weights):
    # Create the game and canvas
    game = Othello()
    canvas = Canvas() if args.visual else None
    # initialize agents 
    agent_b = get_agent(args.black, 'RL_method/model/model_X.pth', 'black', weights)
    agent_w = get_agent(args.white, 'RL_method/model/model_O.pth', 'white', weights)
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
    score = state.getWhiteScore() / (state.getWhiteScore() + state.getBlackScore()) 
    print(f"score:{score*100}%")
    return winner, score


if __name__ == "__main__":
    args = parser.parse_args()
    print('start testing minmax agent against greedy')

    max_score = -999
    best_weights = None

    for n in range(1):
        # w1 = random.uniform(1, 5)
        # w2 = random.uniform(1, 5)
        # w3 = random.uniform(1, 5)
        w1=1
        w2=1
        w3=5
        weights = (w1, w2, w3)
        # run game repeatedly
        wins = 0
        total_score = 0
        for _ in range(50):
            winner, score = play_game(args, weights)
            if winner == "White":
                wins += 1
            total_score += score
        win_rate = wins / 50
        average_score = total_score / 50
        print(f'Win rate for MinmaxAgent: {win_rate * 100}% over {50} games.')
        print(f'average score:{average_score*100}%')
        print(f'using weights: w1={w1}, w2={w2}, w3={w3}')

        if average_score > max_score:
            max_score = average_score
            best_weights = weights
    # print(f'best score:{max_score}')
    # print(f'best weights: w1={best_weights[0]}, w2={best_weights[1]}, w3={best_weights[2]}')
    
    
