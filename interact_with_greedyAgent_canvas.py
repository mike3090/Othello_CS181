from othelloBase import Othello
from greedyAgent import greedyAgent
from canvas import Canvas
import sys
import pygame

# Create an instance of the Othello game
game = Othello()
# Create an instance of the greedy agent
agent = greedyAgent(game)

canvas = Canvas()


# Main game loop
while not game.isEnd():
    # Get the current player's turn
    turn = game.getTurn()

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
        print(f"Greedy Player is thinking...")
        agent.updateGame(game)
        # Let the greedy agent place the piece
        agent.makeMove()

    # Update display 
    canvas.draw_board(game.getBoard(),game.getValidPositions(game.getTurn()))

canvas.game_over(game.getWinner())