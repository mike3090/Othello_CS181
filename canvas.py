import pygame
import tkinter as tk
from tkinter import messagebox
import sys
from othelloBase import Othello

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)

# constants
WIDTH, HEIGHT = 400, 400
SQUARE_SIZE = WIDTH // Othello.SIZE
# Initialize Pygame
pygame.init()

# Set up the display
root = tk.Tk()
root.withdraw()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Othello")

def draw_board(board, validPositions):
    screen.fill(GREEN)
    for row in range(Othello.SIZE):
        for col in range(Othello.SIZE):
            pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)
            if board[row][col] == 1:
                pygame.draw.circle(screen, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)
            elif board[row][col] == -1:
                pygame.draw.circle(screen, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)
    for position in validPositions:
        pygame.draw.circle(screen, RED, (position[1] * SQUARE_SIZE + SQUARE_SIZE // 2, position[0] * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 20)

def game_over(winner):
    if winner:
        # pygame loop here
        messagebox.showinfo("Game Over", f"{['Black', 'White'][winner == -1]} player wins!")
    else:
        messagebox.showinfo("Game Over", "Draw!")

    pygame.quit()
    sys.exit()
    root.lift()

def main():
    OthelloGame = Othello()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                clicked_row = mouse_y // SQUARE_SIZE
                clicked_col = mouse_x // SQUARE_SIZE
                if OthelloGame._checkValid(clicked_row, clicked_col, OthelloGame.getTurn()):
                    OthelloGame.place(clicked_row, clicked_col, OthelloGame.getTurn())
                else:
                    print("Invalid move!")
                    continue
                

        draw_board(OthelloGame.getBoard(),OthelloGame.getValidPositions(OthelloGame.getTurn()))
        pygame.display.flip()

        # game over
        if OthelloGame.isEnd():
            game_over(OthelloGame.getWinner())

if __name__ == "__main__":
    main()
