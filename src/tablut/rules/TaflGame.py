from __future__ import print_function
import sys
#sys.path.append('..')
from tablut.father_class.Game import Game
from tablut.rules.TaflLogic import Board
import numpy as np
from tablut.rules.GameVariants import *
from tablut.utils.Digits import int2base, base2int
from tablut.utils.utils import *



class TaflGame(Game):

    def __init__(self, name):
        self.name = name
        self.getInitBoard()

    def getInitBoard(self):    
        board=Board(Brandubh())
        if self.name=="Brandubh": board=Board(Brandubh())
        if self.name=="ArdRi": board=Board(ArdRi())
        if self.name=="Tablut": board=Board(Tablut())
        if self.name=="Tawlbwrdd": board=Board(Tawlbwrdd())
        if self.name=="Hnefatafl": board=Board(Hnefatafl())
        if self.name=="AleaEvangelii": board=Board(AleaEvangelii())
        self.n=board.size         
        return board
        

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n**4 

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = board.getCopy()
        move = int2base(action,self.n,4)
        b.execute_move(move, player)
        return (b, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        #Note: Ignoreing the passed in player variable since we are not inverting colors for getCanonicalForm and Arena calls with constant 1.
        valids = [0]*self.getActionSize()
        b = board.getCopy()
        legalMoves =  b.get_legal_moves(board.getPlayerToMove())
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x1, y1, x2, y2 in legalMoves:
            valids[base2int([x1, y1, x2, y2], self.n)]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, if player 1 won, -1 if player 1 lost
        return board.done*player

    def getCanonicalForm(self, board, player):
        b = board.getCopy()
        # rules and objectives are different for the different players, so inverting board results in an invalid state.
        return b
    
    # 存储时数据增强
    '''
    def getSymmetries(self, board, pi):
        n = self.n
        if hasattr(board, "astype"):
            base = board.astype(np.int8)
        elif hasattr(board, "getImage"):
            base = np.asarray(board.getImage(), dtype=np.int8)
        else:
            base = np.asarray(board, dtype=np.int8)

        pi = np.asarray(pi, dtype=np.float32)
        perms = action_perms(n)  # (8, n**4)
        out = []
        for k in range(4):
            r = np.rot90(base, k) if base.ndim == 2 else np.rot90(base, k, axes=(-2, -1))
            for flip in (0, 1):
                img = np.fliplr(r) if (flip and base.ndim == 2) else (np.flip(r, axis=-1) if flip else r)
                s = k*2 + flip
                new = pi[perms[s]]           # O(n^4) 的一次性重排
                out.append((img, new))
        return out
    '''
    def getSymmetries(self, board, pi):
        return [(board,pi)]

        # mirror, rotational
        #assert(len(pi) == self.n**4)  
        #pi_board = np.reshape(pi[:-1], (self.n, self.n))
        #l = []

        #for i in range(1, 5):
        #    for j in [True, False]:
        #        newB = np.rot90(board, i)
        #        newPi = np.rot90(pi_board, i)
        #        if j:
        #            newB = np.fliplr(newB)
        #            newPi = np.fliplr(newPi)
        #        l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        #return l

    def stringRepresentation(self, board):
        #print("->",str(board))
        return str(board)

    def getScore(self, board, player):
        if board.done: return 1000*board.done*player
        return board.countDiff(player)



def display(board):
       render_chars = {
             "-1": "b",
              "0": " ",
              "1": "W",
              "2": "K",
             "10": "#",
             "12": "E",
             "20": "_",
             "22": "x",
       }
       print("---------------------")
       image=board.getImage()

       print("  ", " ".join(str(i) for i in range(len(image))))
       for i in range(len(image)-1,-1,-1):
           print("{:2}".format(i), end=" ")

           row=image[i]
           for col in row:
               c = render_chars[str(col)]
               sys.stdout.write(c)
           print(" ") 
       #if (board.done!=0): print("***** Done: ",board.done)  
       print("---------------------")


