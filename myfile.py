import numpy as np
from gobang.GobangGame import GobangGame as Game
from gobang.AlphaBetaSearch import AlphaBetaSearch as Search

g = Game(col=8, row=4, nir=7, defender=-1)

search = Search(g)

myboard = np.zeros((8,4))
#myboard[0][2]=1
res = search.search(myboard, -1, 0, 29)
print(res)
