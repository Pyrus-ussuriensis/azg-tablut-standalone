import numpy as np
import torch
from torch.utils.data import Dataset
from tablut.utils.utils import getNNImage

class RandomSymDataset(Dataset):
    def __init__(self, examples, n, perms):
        self.n = n
        self.perms = perms
        self.items = []
        for b, pi, v, time, size in examples:
            B = b.astype(np.float32) if hasattr(b, "astype") else np.array(b, np.float32)
            self.items.append((
                np.ascontiguousarray(B, dtype=np.float32),   # 基底先存成连续
                np.asarray(pi, dtype=np.float32),
                float(v),
                time,
                size
            ))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        B, pi, v, time, size = self.items[i]
        s = np.random.randint(8)
        k, flip = divmod(s, 2)

        # 旋/翻棋盘（可能产生负stride视图）
        if B.ndim == 2:
            img = np.rot90(B, k)
            if flip: img = np.fliplr(img)
        else:
            img = np.rot90(B, k, axes=(-2, -1))
            if flip: img = np.flip(img, axis=-1)

        # 关键：消除负 stride + 保证 dtype
        img = np.ascontiguousarray(img, dtype=np.float32)  # 或 img = img.copy().astype(np.float32, copy=False)

        # π 用置换表重排；再做连续化
        pi_new = np.ascontiguousarray(pi[self.perms[s]], dtype=np.float32)

        img = getNNImage(img, size, time)

        return (
            torch.from_numpy(img),
            torch.from_numpy(pi_new),
            torch.tensor(v, dtype=torch.float32),
        )

import numpy as np
import torch
from torch.utils.data import Dataset

def scalar_to_planes(B2d: np.ndarray,
                     limit_norm: float = 0.0,
                     extra_terrain_codes=()):
    """
    将标量编码 v = terrain*10 + piece 拆为多平面 (C,H,W).
      piece ∈ {-2,-1,0,+1,+2}; terrain: 0=空,1=王座, 其他码可放在 extra_terrain_codes
    返回顺序: [my_pawn, opp_pawn, my_king, opp_king, throne, (extra terrains...), side, rem_to_limit]
    """
    B2d = np.asarray(B2d, dtype=np.int16)
    H, W = B2d.shape
    terrain = B2d // 10
    piece   = B2d - terrain * 10

    my_pawn  = (piece == +1).astype(np.float32)
    opp_pawn = (piece == -1).astype(np.float32)
    my_king  = (piece == +2).astype(np.float32)
    opp_king = (piece == -2).astype(np.float32)

    throne = (terrain == 1).astype(np.float32)
    planes = [my_pawn, opp_pawn, my_king, opp_king, throne]

    # 可选: 角/城堡等额外地形
    for code in extra_terrain_codes:
        if code == 1: 
            continue  # 避免与 throne 重复
        planes.append((terrain == int(code)).astype(np.float32))

    side_to_move = np.ones((H, W), dtype=np.float32)           # canonical 后恒 1
    rem_to_limit = np.full((H, W), float(limit_norm), dtype=np.float32)

    planes.extend([side_to_move, rem_to_limit])
    return np.stack(planes, axis=0)  # (C,H,W)
