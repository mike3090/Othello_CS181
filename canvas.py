import pygame
import tkinter as tk
from tkinter import messagebox
import sys
from othelloBase import Othello


class Canvas() :
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 128, 0)
    RED = (255, 0, 0)
    # constants
    WIDTH, HEIGHT = 400, 400
    SQUARE_SIZE = WIDTH // Othello.SIZE

    def __init__(self):

        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.root = tk.Tk().withdraw()
        self.screen = pygame.display.set_mode((Canvas.WIDTH, Canvas.HEIGHT))
        pygame.display.set_caption("Othello")

    def draw_board(self, board, validPositions, last_action):
        self.screen.fill(Canvas.GREEN)

        for row in range(Othello.SIZE):
            for col in range(Othello.SIZE):
                pygame.draw.rect(self.screen, Canvas.BLACK, (col * Canvas.SQUARE_SIZE, row * Canvas.SQUARE_SIZE, Canvas.SQUARE_SIZE, Canvas.SQUARE_SIZE), 1)
                if board[row][col] == 1:
                    pygame.draw.circle(self.screen, Canvas.BLACK, (col * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2, row * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2), Canvas.SQUARE_SIZE // 2 - 5)
                elif board[row][col] == -1:
                    pygame.draw.circle(self.screen, Canvas.WHITE, (col * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2, row * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2), Canvas.SQUARE_SIZE // 2 - 5)
    
        for position in validPositions:
            pygame.draw.circle(self.screen, Canvas.RED, (position[1] * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2, position[0] * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2), Canvas.SQUARE_SIZE // 2 - 20)
        
        if last_action is not None:
            pygame.draw.circle(self.screen, Canvas.GREEN, (last_action[1] * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2, last_action[0] * Canvas.SQUARE_SIZE + Canvas.SQUARE_SIZE // 2), Canvas.SQUARE_SIZE // 2 - 10)
        # update display
        pygame.display.flip()

    def game_over(self, winner):
        if winner:
            # pygame loop here
            messagebox.showinfo("Game Over", f"{['Black', 'White'][winner == -1]} player wins!")
        else:
            messagebox.showinfo("Game Over", "Draw!")

        pygame.quit()

        sys.exit()
