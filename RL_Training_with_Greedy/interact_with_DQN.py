import sys
# sys.path.append("..")
from oldothelloBase import Othello
from DQN_Agent import DQNAgent
from canvas import Canvas
import pygame

# 用户想先手，就把他改成True，否则改成False
if_user_first = False


# Create an instance of the Othello game
game = Othello()
# Create an instance of the greedy agent
if if_user_first:
    agent = DQNAgent(game, -1)  # 可以修改成DuelingDQN
else:
    agent = DQNAgent(game, 1)

canvas = Canvas()


# Main game loop
while not game.isEnd():
    # Get the current player's turn
    turn = game.getTurn()

    if if_user_first:
        if turn == 1:
            # Take action 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    clicked_row = mouse_y // Canvas.SQUARE_SIZE
                    clicked_col = mouse_x // Canvas.SQUARE_SIZE
                    if game._checkValid(clicked_row, clicked_col, game.getTurn()):
                        game.place(clicked_row, clicked_col, game.getTurn())
                    else:
                        print("Invalid move!")
                        continue
        else:
            print(f"DQN Player is thinking...")
            agent.updateGame(game)
            # Let the DQN agent place the piece
            agent.makeMove()
    else:
        if turn == 1:
            print(f"DQN Player is thinking...")
            agent.updateGame(game)
            # Let the DQN agent place the piece
            agent.makeMove()
        else:
            # Take action 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    clicked_row = mouse_y // Canvas.SQUARE_SIZE
                    clicked_col = mouse_x // Canvas.SQUARE_SIZE
                    if game._checkValid(clicked_row, clicked_col, game.getTurn()):
                        game.place(clicked_row, clicked_col, game.getTurn())
                    else:
                        print("Invalid move!")
                        continue

    # Update display 
    canvas.draw_board(game.getBoard(),game.getValidPositions(game.getTurn()))

canvas.game_over(game.getWinner())