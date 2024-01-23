from oldothelloBase import Othello
from DQN_Agent import DQNAgent
from DQN_Agent import DQNAgent_5000
from DQN_Agent import DQNAgent_460000
from oldgreedyAgent import greedyAgent

# 用户想先手，就把他改成True，否则改成False
if_DQN_first = False 

# # Create an instance of the Othello game
# game = Othello()
# # Create an instance of the greedy agent
# if if_DQN_first:
#     print("X IS DQN")
#     agentX = DQNAgent(game, 1)
#     agentO = greedyAgent(game)
# else:
#     print("X IS Greedy")
#     agentX = greedyAgent(game)
#     agentO = DQNAgent(game, -1)

# canvas = Canvas()


# # Main game loop
# while not game.isEnd():
#     # Get the current player's turn
#     turn = game.getTurn()
#     playerName = "X" if turn == 1 else "O"

#     if not if_DQN_first:
#         if turn == 1:
#             #print(f"Greedy Player {playerName} is thinking...")
#             agentX.updateGame(game)
#             # Let the greedy agent place the piece
#             agentX.makeMove()
#         else:
#             #print(f"DQN Player is thinking...")
#             agentO.updateGame(game)
#             # Let the DQN agent place the piece
#             agentO.makeMove()
#     else:
#         if turn == 1:
#             #print(f"DQN Player is thinking...")
#             agentX.updateGame(game)
#             # Let the DQN agent place the piece
#             agentX.makeMove()
#         else:
#             #print(f"Greedy Player {playerName} is thinking...")
#             agentO.updateGame(game)
#             # Let the greedy agent place the piece
#             agentO.makeMove()

#     # Update display 
#     canvas.draw_board(game.getBoard(),game.getValidPositions(game.getTurn()))

# canvas.game_over(game.getWinner())

win_cnt = 0
for i in range(500):
    if i%10 == 0:
        print(i)
    # Create an instance of the Othello game
    game = Othello()
    # Create an instance of the greedy agent
    if if_DQN_first:
        # print("X IS DQN")
        # agentX = DQNAgent_5000(game, 1)
        agentO = greedyAgent(game)
        agentX = DQNAgent_460000(game, 1)
    else:
        # print("X IS Greedy")
        agentO = DQNAgent_460000(game, -1)
        # agentO = DQNAgent_5000(game, -1)
        agentX = greedyAgent(game)

    while not game.isEnd():
        # Get the current player's turn
        turn = game.getTurn()
        playerName = "X" if turn == 1 else "O"

        if not if_DQN_first:
            if turn == 1:
                #print(f"Greedy Player {playerName} is thinking...")
                agentX.updateGame(game)
                # Let the greedy agent place the piece
                agentX.makeMove()
            else:
                #print(f"DQN Player is thinking...")
                agentO.updateGame(game)
                # Let the DQN agent place the piece
                agentO.makeMove()
        else:
            if turn == 1:
                #print(f"DQN Player is thinking...")
                agentX.updateGame(game)
                # Let the DQN agent place the piece
                agentX.makeMove()
            else:
                #print(f"Greedy Player {playerName} is thinking...")
                agentO.updateGame(game)
                # Let the greedy agent place the piece
                agentO.makeMove()

    # canvas.draw_board(game.getBoard(),game.getValidPositions(game.getTurn()))

    winner = game.getWinner()
    winnerName = "X" if winner == 1 else "O"
    print(winnerName)
    if if_DQN_first and winner == 1:
        win_cnt+=1
    elif not if_DQN_first and winner == -1:
        win_cnt+=1
    
if if_DQN_first:
    print("X is DQN")
else:
    print("O is DQN")
print(f"460000 Win Rate: {win_cnt}/500")