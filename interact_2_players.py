from othelloBase import Othello
# Create an instance of the Othello game
game = Othello()

# Main game loop
while not game.isEnd():
    # Print the current board
    print("Current Board:")
    game.printBoard()
    print()

    # Get the current player's turn
    turn = game.getTurn()
    playerName = "X" if turn == 1 else "O"

    # Get the valid positions for the current player
    valid_positions = game.getValidPositions(turn)

    # Prompt the current player to enter their move
    print("\033[0;32;40mYour possible moves: \033[0m" + str(valid_positions))
    print(f"Player {playerName}, enter your move \033[0;31;40m(row, col) / (第几行，第几列)\033[0m:")
    x = int(input("row = "))
    y = int(input("col = "))

    if x == 9: # test undo
        game.undo()
        continue
    # Check if the entered move is valid
    elif (x, y) in valid_positions:
        # Place the player's piece on the board
        game.place(x, y, turn)
    else:
        print("\033[0;31;40mInvalid move. Try again.\033[0m")

# Print the final board
game.printBoard()

# Get the winner and score
winner = game.getWinner()
score = game.getScore()

# Print the winner and score
if winner == 1:
    print("Black wins!")
elif winner == -1:
    print("White wins!")
else:
    print("It's a tie!")
print(f"Score for X: {score}")

