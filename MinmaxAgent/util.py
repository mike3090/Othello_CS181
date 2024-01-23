import sys
import inspect

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