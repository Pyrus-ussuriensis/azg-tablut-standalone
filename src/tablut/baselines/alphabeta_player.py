import numpy as np
from tablut.baselines.greedy_player import _eval_board
from tablut.utils.Digits import base2int

PASS = lambda n: n**4 - 1

class AlphaBetaTaflPlayer:
    def __init__(self, game, depth=2):
        self.game = game
        self.depth = depth

    def startGame(self): pass
    def endGame(self):   pass

    # 工具函数
    def _moves(self, board, cur):
        raw = board.get_legal_moves(cur)              # [(x1,y1,x2,y2),...]
        n = self.game.n
        return [base2int([x1,y1,x2,y2], n) for (x1,y1,x2,y2) in raw]

    def _term_origin(self, b, origin):
        r = self.game.getGameEnded(b, origin)         # 直接以 origin 视角判终局
        return None if r == 0 else r * 1e6

    def _eval_origin(self, b, origin):
        bc = self.game.getCanonicalForm(b, origin)    # 规范到 origin 视角
        return _eval_board(self.game, bc)             # bc中传入了我们的视角评估

    # αβ
    def _ab(self, board, cur, origin, d, alpha, beta):
        tv = self._term_origin(board, origin)
        if tv is not None: # 终局 origin 视角值
            return tv, None
        if d == 0: # 叶子评估 origin 视角值
            return self._eval_origin(board, origin), None

        pool = self._moves(board, cur)
        if not pool:
            # 保险：无子可走应已由终局捕获
            return self._term_origin(board, origin) or -1e9, None

        best_a = None
        if cur == origin: # MAX（同阵营）
            v = -1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _ = self._ab(nb, nxt, origin, d-1, alpha, beta)
                if sv > v: v, best_a = sv, int(a)
                if v > alpha: alpha = v
                if alpha >= beta: break
            return v, best_a
        else: # MIN（对手阵营）
            v =  1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _ = self._ab(nb, nxt, origin, d-1, alpha, beta)
                if sv < v: v, best_a = sv, int(a)
                if v < beta: beta = v
                if alpha >= beta: break
            return v, best_a

    def __call__(self, board):
        origin = board.getPlayerToMove()
        _, a = self._ab(board, origin, origin, self.depth, -1e18, 1e18)
        if a is None:
            ms = self._moves(board, origin)
            return int(ms[0]) if ms else 0
        return int(a)

