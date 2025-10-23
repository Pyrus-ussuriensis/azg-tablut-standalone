class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]



def rot_xy(x, y, n, k):
    k %= 4
    if k == 0: return x, y
    if k == 1: return n-1-y, x
    if k == 2: return n-1-x, n-1-y
    if k == 3: return y, n-1-x
    return x, y

import numpy as np
from functools import lru_cache

def _rot_xy_vec(x, y, n, k):
    k %= 4
    if k == 0: return x, y
    if k == 1: return n-1-y, x
    if k == 2: return n-1-x, n-1-y
    return y, n-1-x

@lru_cache(maxsize=None)
def action_perms(n: int) -> np.ndarray:
    # 返回 shape=(8, n**4) 的置换表；顺序为 s=k*2+flip，pass 槽位保持不变
    a  = np.arange(n**4)
    x1 =  a % n
    y1 = (a // n)    % n
    x2 = (a // n**2) % n
    y2 = (a // n**3) % n

    perms = np.empty((8, n**4), dtype=np.int64)
    for k in range(4):
        rx1, ry1 = _rot_xy_vec(x1, y1, n, k)
        rx2, ry2 = _rot_xy_vec(x2, y2, n, k)
        for flip in (0, 1):
            fx1 = n-1-rx1 if flip else rx1
            fx2 = n-1-rx2 if flip else rx2
            new_idx = fx1 + ry1*n + fx2*(n**2) + ry2*(n**3)
            s = k*2 + flip
            perms[s] = new_idx
            perms[s, -1] = n**4 - 1    
    return perms
