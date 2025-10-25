
# Note: Run this file from Arena directory (the one above /tafl)

from tablut import Arena
from tablut.models.MCTS import MCTS
from tablut.rules.TaflGame import TaflGame, display
from tablut.rules.TaflPlayers import *
#from tafl.keras.NNet import NNetWrapper as NNet

import numpy as np
from tablut.utils.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TaflGame("Tablut")

# all players
rp = RandomTaflPlayer(g).play
gp = GreedyTaflPlayer(g).play
hp = HumanTaflPlayer(g).play

from tablut.baselines.alphabeta_player import AlphaBetaTaflPlayer
from tablut.baselines.greedy_player import GreedyTaflPlayer
from tablut.baselines.random_player import RandomPlayer

# nnet players
#n1 = NNet(g)
#n1.load_checkpoint('./pretrained_models/tafl/keras/','6x100x25_best.pth.tar')
#args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
a, b, c = AlphaBetaTaflPlayer(g,2), GreedyTaflPlayer(g), RandomPlayer(g)

#arena = Arena.Arena(a, c, g, display=display)
arena = Arena.Arena(a, b, g)

n = 10
#arena = Arena.Arena(a, b, g, display=display)
print(arena.playGames(n, verbose=False))
#print(arena.playGames(100, verbose=True))
arena = Arena.Arena(a, c, g)
#arena = Arena.Arena(gp, rp, g, display=display)
print(arena.playGames(n, verbose=False))
arena = Arena.Arena(b, c, g)
print(arena.playGames(n, verbose=False))
#print(arena.playGames(100, verbose=True))
