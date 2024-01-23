from util import *
from othelloBase import GameState
from othelloBase import Agent
import pygame
from canvas import Canvas

class MouseAgent(Agent):
    '''
    implement a mouse agent controlled by humans
    '''
    def getAction(self, state: GameState):
        '''
        Find click position 
        '''
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    clicked_row = mouse_y // Canvas.SQUARE_SIZE
                    clicked_col = mouse_x // Canvas.SQUARE_SIZE
                    if state._checkValid(clicked_row, clicked_col):
                        return clicked_row, clicked_col