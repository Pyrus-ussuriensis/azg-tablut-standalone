import numpy as np

PASS = lambda n: n**4 - 1
# ============== 公共启发式（对白/王方，分数越大越好） ==============
def _eval_board(game, b) -> float:
    img = b.getImage()
    n = len(img)

    # 找王
    king = None
    for x, y, t in b.pieces:
        if t == 2:
            king = (x, y); break
    if king is None:
        return -1e9

    x, y = king
    dist_edge = min(x, y, n-1-x, n-1-y)

    # 子力差（白含王）
    w = sum(1 for _, _, t in b.pieces if t > 0)
    bcnt = sum(1 for _, _, t in b.pieces if t < 0)

    # 王机动性（四向可走步数）
    mob = 0
    occ = {(px, py) for px, py, _ in b.pieces}
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        cx, cy = x, y
        while True:
            nx, ny = cx+dx, cy+dy
            if nx < 0 or ny < 0 or nx >= n or ny >= n: break
            if (nx, ny) in occ: break
            mob += 1; cx, cy = nx, ny

    return -1.5*dist_edge + 0.2*mob + 0.05*(w - bcnt)

# ===================== 一层贪心 =====================
class GreedyTaflPlayer:
    def __init__(self, game):
        self.game = game

    def startGame(self):  # Arena 每局开头会调用（可选）
        pass

    def endGame(self):    # Arena 每局结束会调用（可选）
        pass

    def __call__(self, canonicalBoard):
        n = self.game.n
        valids = self.game.getValidMoves(canonicalBoard, 1)
        legal = np.where(valids == 1)[0]
        if len(legal) == 0:
            return PASS(n)

        # 能走就不选 pass
        pool = legal[legal != PASS(n)]
        if len(pool) == 0:
            pool = legal

        best, best_as = -1e18, []
        for a in pool:
            nb, _ = self.game.getNextState(canonicalBoard, 1, int(a))
            sc = _eval_board(self.game, nb)
            if sc > best + 1e-9:
                best, best_as = sc, [a]
            elif abs(sc - best) <= 1e-9:
                best_as.append(a)

        return int(np.random.choice(best_as)) if best_as else int(pool[0])

