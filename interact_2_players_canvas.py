from othelloBase import Othello
from canvas import Canvas
import sys
import pygame
# Create an instance of the Othello game
game = Othello()

canvas = Canvas()

# Main game loop
while not game.isEnd():

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

