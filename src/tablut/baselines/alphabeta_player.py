import numpy as np
from tablut.baselines.greedy_player import _eval_board

PASS = lambda n: n**4 - 1

# ===================== Alpha-Beta =====================
class AlphaBetaTaflPlayer:
    def __init__(self, game, depth=2):
        self.game = game
        self.depth = depth

    def startGame(self): pass
    def endGame(self):   pass

    def _term_val(self, b, cur):
        # 终局优先：有 getGameEnded 用它，没有就用 Board._getWinLose()
        if hasattr(self.game, "getGameEnded"):
            r = self.game.getGameEnded(b, cur)   # 1 白胜, -1 白负, 0 未终
            if r != 0:
                return ( r if cur == 1 else -r) * 1e6
        if hasattr(b, "_getWinLose"):
            r = b._getWinLose()                  # 1 白胜, -1 白负, 0 未终
            if r != 0:
                return ( r if cur == 1 else -r) * 1e6
        return None

    def _ab(self, board, cur, d, alpha, beta):
        tv = self._term_val(board, cur)
        if tv is not None: return tv, None
        if d == 0:         return _eval_board(self.game, board), None

        valids = self.game.getValidMoves(board, cur)
        legal  = np.where(valids == 1)[0]
        if len(legal) == 0:
            return _eval_board(self.game, board), PASS(self.game.n)

        pool = legal[legal != PASS(self.game.n)]
        if len(pool) == 0: pool = legal

        best_a = None
        if cur == 1:  # MAX（白）
            v = -1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _   = self._ab(nb, nxt, d-1, alpha, beta)
                if sv > v: v, best_a = sv, int(a)
                alpha = max(alpha, v)
                if alpha >= beta: break
            return v, best_a
        else:         # MIN（黑）
            v =  1e18
            for a in pool:
                nb, nxt = self.game.getNextState(board, cur, int(a))
                sv, _   = self._ab(nb, nxt, d-1, alpha, beta)
                if sv < v: v, best_a = sv, int(a)
                beta = min(beta, v)
                if alpha >= beta: break
            return v, best_a

    def __call__(self, canonicalBoard):
        _, a = self._ab(canonicalBoard, 1, self.depth, -1e18, 1e18)
        if a is None:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            legal  = np.where(valids == 1)[0]
            return int(legal[0]) if len(legal) else PASS(self.game.n)
        return int(a)

'''
class AlphaBetaTaflPlayer:
    def __init__(self, game, depth=3):
        self.game, self.depth = game, depth
        self.TT = {}

    def startGame(self): self.TT.clear()
    def endGame(self):   pass

    def _term_val(self, b):
        if hasattr(self.game, "getGameEnded"):
            r = self.game.getGameEnded(b, 1)
            if r != 0: return r * 1e6
        if hasattr(b, "_getWinLose"):
            r = b._getWinLose()
            if r != 0: return r * 1e6
        return None

    def _ab(self, board, cur, d, alpha, beta):
        key = (self.game.stringRepresentation(board), cur, d)
        if key in self.TT: return self.TT[key]

        tv = self._term_val(board)
        if tv is not None:
            self.TT[key] = (tv, None); return self.TT[key]
        if d == 0:
            v = _eval_board(self.game, board)
            self.TT[key] = (v, None); return self.TT[key]

        valids = self.game.getValidMoves(board, cur)
        legal  = np.where(valids==1)[0]
        pool   = legal[legal != PASS(self.game.n)]
        if len(pool)==0: pool = legal

        # —— 走法排序 —— #
        moves = []
        for a in pool:
            nb, nxt = self.game.getNextState(board, cur, int(a))
            h = _eval_board(self.game, nb)     # 白方视角
            moves.append((h, a, nb, nxt))
        moves.sort(reverse=(cur==1))           # MAX: 大到小；MIN: 小到大

        best_a = None
        if cur == 1:    # MAX
            v = -1e18
            for h, a, nb, nxt in moves:
                sv, _ = self._ab(nb, nxt, d-1, alpha, beta)
                if sv > v: v, best_a = sv, int(a)
                alpha = max(alpha, v)
                if alpha >= beta: break
        else:           # MIN
            v =  1e18
            for h, a, nb, nxt in moves:
                sv, _ = self._ab(nb, nxt, d-1, alpha, beta)
                if sv < v: v, best_a = sv, int(a)
                beta = min(beta, v)
                if alpha >= beta: break

        self.TT[key] = (v, best_a)
        return self.TT[key]

    def __call__(self, canonicalBoard):
        v, a = self._ab(canonicalBoard, 1, self.depth, -1e18, 1e18)
        if a is None:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            legal  = np.where(valids==1)[0]
            return int(legal[0]) if len(legal) else PASS(self.game.n)
        return int(a)
'''


'''
# baselines_players.py 片段
import time
import numpy as np

def PASS(n: int) -> int:
    return n**4 - 1

def decode_move(a: int, n: int):
    x1 = a % n; a //= n
    y1 = a % n; a //= n
    x2 = a % n; a //= n
    y2 = a % n
    return x1, y1, x2, y2

# —— 轻量启发式（对白/王方，分数越大越好）——
def eval_board(game, b) -> float:
    img = b.s()
    n = len(img)
    king = None
    w_cnt = 0; b_cnt = 0
    occ = set()
    for x, y, t in b.pieces:
        occ.add((x, y))
        if t > 0: w_cnt += 1
        elif t < 0: b_cnt += 1
        if t == 2: king = (x, y)
    if king is None: return -1e9
    x, y = king
    dist_edge = min(x, y, n-1-x, n-1-y)
    # 限距机动性（快一点）
    mob = 0
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        cx, cy = x, y
        for _ in range(4):
            nx, ny = cx+dx, cy+dy
            if nx<0 or ny<0 or nx>=n or ny>=n or (nx,ny) in occ: break
            mob += 1; cx, cy = nx, ny
    return -1.6*dist_edge + 0.22*mob + 0.05*(w_cnt - b_cnt)

class AlphaBetaTaflPlayer:
    def __init__(self, game, depth=3, root_topk=14, inner_topk=9,
                 lmr_after=999, lmr_cut=0,  # 先关掉浅层 LMR：更稳
                 verbose=1):
        self.game = game
        self.depth = depth
        self.root_topk = root_topk
        self.inner_topk = inner_topk
        self.lmr_after = lmr_after
        self.lmr_cut = lmr_cut
        self.verbose = verbose

        self.TT = {}         # (stringRep, cur, d) -> (value, best_a, info)
        self.TT_best = {}    # (stringRep, cur) -> best_a  （排序提示）
        self.killers = {}    # killers[ply] = {a1, a2}
        self.HE = {}         # 评估缓存：用图像做 key

        self.reset_stats()

    # ===== Arena hooks =====
    def startGame(self):
        self.TT.clear(); self.TT_best.clear(); self.killers.clear(); self.HE.clear()
        self.reset_stats()
        self._t_game = time.perf_counter()

    def endGame(self):
        t = time.perf_counter() - self._t_game
        if self.verbose >= 1:
            print(f"[AB] game_done: nodes={self.stat_nodes_total}, "
                  f"eval_calls={self.stat_eval_calls}, he_hits={self.stat_he_hits}, "
                  f"tt_hits={self.stat_tt_hits}, cut(M={self.stat_cut_max}/m={self.stat_cut_min}), "
                  f"time={t:.3f}s")

    def __call__(self, board):
        t0 = time.perf_counter()
        self.stat_nodes_move = 0; self.stat_tt_hits_move = 0
        self.stat_cut_max_move = 0; self.stat_cut_min_move = 0

        v, a, info = self._ab(board, cur=1, d=self.depth,
                              alpha=-1e18, beta=1e18, is_root=True, ply=0)

        dt = time.perf_counter() - t0
        self.stat_nodes_total += self.stat_nodes_move
        self.stat_tt_hits += self.stat_tt_hits_move
        self.stat_cut_max += self.stat_cut_max_move
        self.stat_cut_min += self.stat_cut_min_move

        if self.verbose >= 1:
            k0 = info.get("k0", None)
            print(f"[AB] move: bestA={a}, val={v:.2f}, nodes={self.stat_nodes_move}, "
                  f"tt_hits={self.stat_tt_hits_move}, cuts(M={self.stat_cut_max_move}/m={self.stat_cut_min_move}), "
                  f"K0={k0}, time={dt:.3f}s")
        if self.verbose >= 2:
            root_list = info.get("root_moves", [])
            dec = [(h, decode_move(a, self.game.n)) for (h, a) in root_list[:10]]
            print("[AB] root order (h, x1,y1,x2,y2) top:", dec)

        if a is None:
            valids = self.game.getValidMoves(board, 1)
            legal = np.where(valids == 1)[0]
            return int(legal[0]) if len(legal) else PASS(self.game.n)
        return int(a)

    # ===== internals =====
    def reset_stats(self):
        self.stat_nodes_total = self.stat_tt_hits = 0
        self.stat_cut_max = self.stat_cut_min = 0
        self.stat_eval_calls = self.stat_he_hits = 0
        self.stat_nodes_move = self.stat_tt_hits_move = 0
        self.stat_cut_max_move = self.stat_cut_min_move = 0

    def _term_val(self, b, cur):
        """
        统一“对白方”的分数：
          r = game.getGameEnded(b, cur):  1 表示“cur 赢了”，-1 表示“cur 输了”
          -> 白方分数 = r (cur==1 时)；= -r (cur==-1 时)
        """
        if hasattr(self.game, "getGameEnded"):
            r = self.game.getGameEnded(b, cur)
            if r != 0:
                return (r * 1e6) if cur == 1 else (-r * 1e6)
        if hasattr(b, "_getWinLose"):
            r = b._getWinLose()  # 1白胜,-1白负,0未终
            if r != 0:
                return r * 1e6
        return None

    def _he_key(self, b):
        # 只取棋盘图像，忽略“行棋方”等易变化字段，提高命中
        img = b.getImage()
        try:
            return img.tobytes()
        except Exception:
            # 回退：把 ndarray 展平成 tuple
            return tuple(map(int, np.asarray(img).ravel()))

    def _he_eval(self, b):
        k = self._he_key(b)
        v = self.HE.get(k)
        if v is None:
            v = eval_board(self.game, b)
            self.HE[k] = v
            self.stat_eval_calls += 1
        else:
            self.stat_he_hits += 1
        return v

    def _children_sorted(self, board, cur, limit, ply):
        valids = self.game.getValidMoves(board, cur)
        legal = np.where(valids == 1)[0]
        if len(legal) == 0: return []

        pool = legal[legal != PASS(self.game.n)]
        if len(pool) == 0: pool = legal

        sr = self.game.stringRepresentation(board)
        best_hint = self.TT_best.get((sr, cur))
        killers = self.killers.get(ply, set())

        primaries, secondaries = [], []
        for a in pool:
            nb, nxt = self.game.getNextState(board, cur, int(a))
            h = self._he_eval(nb)
            item = (h, int(a), nb, nxt)
            if a == best_hint or a in killers:
                primaries.append(item)
            else:
                secondaries.append(item)

        primaries.sort(reverse=(cur == 1))
        secondaries.sort(reverse=(cur == 1))
        moves = primaries + secondaries
        if limit is not None and len(moves) > limit:
            moves = moves[:limit]
        return moves

    def _record_killer(self, ply, a):
        ks = self.killers.setdefault(ply, set())
        if a not in ks:
            ks.add(a)
            if len(ks) > 2:
                ks.pop()

    def _ab(self, board, cur, d, alpha, beta, is_root, ply):
        self.stat_nodes_move += 1

        key = (self.game.stringRepresentation(board), cur, d)
        if key in self.TT:
            self.stat_tt_hits_move += 1
            return self.TT[key]

        tv = self._term_val(board, cur)
        if tv is not None:
            out = (tv, None, {})
            self.TT[key] = out
            self.TT_best[(key[0], cur)] = None
            return out

        if d == 0:
            v = self._he_eval(board)
            out = (v, None, {})
            self.TT[key] = out
            return out

        limit = self.root_topk if is_root else self.inner_topk
        moves = self._children_sorted(board, cur, limit, ply)
        if not moves:
            v = self._he_eval(board)
            out = (v, PASS(self.game.n), {"k0": 0} if is_root else {})
            self.TT[key] = out
            return out

        best_a = None
        info = {}
        if is_root:
            info["k0"] = len(moves)
            info["root_moves"] = [(round(h, 2), a) for (h, a, *_ ) in moves]

        if cur == 1:  # MAX（白）
            v = -1e18
            for i, (h, a, nb, nxt) in enumerate(moves):
                dd = d - 1  # 先关闭 LMR（上面构造参数 lmr_cut=0）
                sv, _, _ = self._ab(nb, nxt, dd, alpha, beta, False, ply+1)
                if sv > v:
                    v, best_a = sv, a
                alpha = max(alpha, v)
                if alpha >= beta:
                    self.stat_cut_max_move += 1
                    self._record_killer(ply, a)
                    break
        else:         # MIN（黑）
            v = 1e18
            for i, (h, a, nb, nxt) in enumerate(moves):
                dd = d - 1
                sv, _, _ = self._ab(nb, nxt, dd, alpha, beta, False, ply+1)
                if sv < v:
                    v, best_a = sv, a
                beta = min(beta, v)
                if alpha >= beta:
                    self.stat_cut_min_move += 1
                    self._record_killer(ply, a)
                    break

        self.TT_best[(key[0], cur)] = best_a
        out = (v, best_a, info)
        self.TT[key] = out
        return out
'''