import sys
import inspect
from othelloBase import GameState

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def printBoard(board):
    '''
    Return a string representation of the board.
    '''
    print(
        '\n'.join(["  0 1 2 3 4 5 6 7"]+[str(i)+" "+(' '.join(map(lambda x: 'X' if x == 1 else 'O' if x == -1 else '.', board[i]))) for i in range(len(board))])
    )

def scoreEvaluationFunction(currentGameState: GameState, color):
    """
    This evaluation function is composed of:
    1. how many pieces does the chosen color own more than its opponent
    2. how many edge pieces does the chosen color own
    3. how many corner pieces does the chosen color own
    """

    # feature 1
    w = currentGameState.getWhiteScore()
    b = currentGameState.getBlackScore()
    if color == 'black':
        score_1 = b - w
        index = 1
    elif color == 'white':
        score_1 = w - b
        index = -1
    
    # feature 2
    safe_points = set()

    for point in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        if currentGameState.board[point[0]][point[1]] == index:
            safe_points.add(point)
    corner_num = len(safe_points)

    def find_connected_points(currentGameState, point, index, direction): # recursive search function, search for connected parts of corner points
        connected_points = []
        next_point = (point[0] + direction[0], point[1] + direction[1])
        if 0 <= next_point[0] < 8 and 0 <= next_point[1] < 8:
            if currentGameState.board[next_point[0]][next_point[1]] == index:
                connected_points.append(next_point)
                connected_points += find_connected_points(currentGameState, next_point, index, direction)
        return connected_points
    
    if (0,0) in safe_points:
        safe_points.update(find_connected_points(currentGameState, (0, 0), index, (1, 0)))
        safe_points.update(find_connected_points(currentGameState, (0, 0), index, (0, -1)))

    if (0,7) in safe_points:
        safe_points.update(find_connected_points(currentGameState, (0, 7), index, (-1, 0)))
        safe_points.update(find_connected_points(currentGameState, (0, 7), index, (0, -1)))

    if (7,0) in safe_points:
        safe_points.update(find_connected_points(currentGameState, (7, 0), index, (1, 0)))
        safe_points.update(find_connected_points(currentGameState, (7, 0), index, (0, 1)))

    if (7,7) in safe_points:
        safe_points.update(find_connected_points(currentGameState, (7, 7), index, (-1, 0)))
        safe_points.update(find_connected_points(currentGameState, (7, 7), index, (0, 1)))

    score_2 = len(safe_points) - corner_num

    # feature 3
    score_3 = 0
    for point in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        if currentGameState.board[point[0]][point[1]] == index:
            score_3 += 1
        if currentGameState.board[point[0]][point[1]] == -index:
            score_3 -= 1
    
    # weights for each score
    w1=2.833337761398725
    w2=3.660558028419165
    w3=3.636812042078707
    return w1* score_1 + w2 * score_2 + w3* score_3
