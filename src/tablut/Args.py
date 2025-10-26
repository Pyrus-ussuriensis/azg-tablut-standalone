from tablut.utils.utils import *
args0 = dotdict({
    'numIters': 1000,
    'numEps': 64,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 20,
    'experiment': 0,
    'evaluate': 128

})

args1 = dotdict({
    'numIters': 60,
    'numEps': 80,
    #'numEps': 200,
    'tempThreshold': 10,
    'updateThreshold': 0.58,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 150,
    #'numMCTSSims': 200,
    'arenaCompare': 50,
    #'arenaCompare': 200,
    'cpuct': 1.5,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 10,
    'experiment': 0,

})

# 方案B｜标准（更强，时间更久）
args2 = dotdict({
  'numIters': 100,
  'numEps': 500,
  'tempThreshold': 10,
  'updateThreshold': 0.60,
  'maxlenOfQueue': 500_000,
  'numMCTSSims': 400,
  'arenaCompare': 400,
  'cpuct': 1.25,
  'checkpoint': './temp/',
  'load_model': True,
  'load_folder_file': 'best.pth.tar',
  'numItersForTrainExamplesHistory': 15,
  'experiment': 1,
})

args3 = dotdict({
    'numIters': 400,                 # 迭代上限足够大；实际早停或以best替换为准
    'numEps': 64,                    # 每迭代自博弈局数（多进程并行友好）
    'tempThreshold': 10,             # 前10步用温度>0，多样化数据
    'updateThreshold': 0.55,         # 早期门槛稍放宽（详见下方分段门槛）
    'maxlenOfQueue': 200000,         # 经验缓冲区大小合适
    'numMCTSSims': 64,               # 自博弈MCTS模拟次数（批量前向+AMP即可承受）
    'arenaCompare': 64,              # 快速筛查对弈局数（换边）
    'cpuct': 1.5,                    # 常用稳定值，探索/利用更均衡

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'numItersForTrainExamplesHistory': 10,  # 历史样本保留10轮，防止策略陈旧
    'experiment': 0,

    # 仅评测用：与基准/AB打确认赛的局数（换边）
    'evaluate': 128
})

args4 = dotdict({
    # —— 迭代与数据 —— #
    'numIters': 300,
    'numEps': 128,                  # ↑ 自博弈局数：直接抬高训练时长与数据量
    'numItersForTrainExamplesHistory': 10,
    'maxlenOfQueue': 200000,

    # —— MCTS（训练/自博弈） —— #
    'numMCTSSims': 128,             # ↑ 主力模拟数
    'playout_cap': [64, 96, 128],   # 每步随机取一个（PCR），稳住数据多样性
    'cpuct': 1.5,
    'tempThreshold': 10,
    'dirichlet_alpha': 0.25,
    'dirichlet_eps': 0.25,
    'resign_enable_after': 6,
    'resign_threshold': -0.85,

    # —— 评测/门控（对上一代快筛） —— #
    'arenaCompare': 48,             # ↓ 快筛盘数，缩短评测
    'updateThreshold': 0.55,        # 建议分段门槛：1–5轮0.52，6–15轮0.55，16+轮0.58

    # —— 与 AlphaBeta 的确认赛（仅快筛通过才触发） —— #
    'evaluate': 96,                 # ↓ 确认赛盘数，减少总评测时间
    'eval_ab_every': 8,             # 仅每8轮做一次 AB 基准确认，其他轮只与上一代比
    'ab_depth': 2,
    'disable_dirichlet_in_eval': True,
    'eval_temperature_zero': True,

    # —— 训练超参（若你在 Trainer 里读取） —— #
    'epochs': 8,                    # ↑ 每轮训练 epoch，提升每轮“学习量”
    'batch_size': 128,
    'ema_decay': 0.999,             # 若实现 EMA，评测用 EMA 权重

    # —— 存档/杂项 —— #
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 0,
    'in_planes': 6,
})

args5 = dotdict({
    # 迭代与数据
    'numIters': 400,
    'numEps': 64,
    'numItersForTrainExamplesHistory': 10,
    'maxlenOfQueue': 200000,

    # MCTS
    'numMCTSSims': 64,        # 阶段A手动设32；阶段B及后 64
    'cpuct': 1.5,
    'tempThreshold': 10,

    # 评测/门控
    'arenaCompare': 64,       # 快筛
    'evaluate': 128,          # 确认赛（通过快筛才打）
    'updateThreshold': 0.55,  # 分段：A=0.52, B=0.55, C=0.58

    # 模型/存档
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 0,

    # —— 建议新增的开关（在 Coach/MCTS 里读到就生效）——
    'dirichlet_alpha': 0.25,      # 根注噪
    'dirichlet_eps': 0.25,
    'playout_cap': [32,64,96],    # A 阶段用 [16,32,64]
    'resign_enable_after': 5,
    'resign_threshold': -0.85,
    'ema_decay': 0.999,           # 若实现了EMA，用其做评测
    'disable_dirichlet_in_eval': True,
    'eval_temperature_zero': True,
})


# 测试跑通使用：
args6 = dotdict({
    # 迭代与数据
    'numIters': 400,
    'numEps': 100,
    'numItersForTrainExamplesHistory': 10,
    'maxlenOfQueue': 200000,

    # MCTS
    'numMCTSSims':100,        # 阶段A手动设32；阶段B及后 64
    'cpuct': 1.5,
    'tempThreshold': 10,

    # 评测/门控
    'arenaCompare': 50,       # 快筛
    'evaluate': 100,          # 确认赛（通过快筛才打）
    'updateThreshold': 0.55,  # 分段：A=0.52, B=0.55, C=0.58

    # 模型/存档
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 0,

    # —— 建议新增的开关（在 Coach/MCTS 里读到就生效）——
    'dirichlet_alpha': 0.25,      # 根注噪
    'noise_eps': 0.25,
})

# 自博弈/评测&搜索
args = dotdict({
    # 迭代与数据 —— 维持即可
    'numIters': 400,
    'numEps': 100,
    'numItersForTrainExamplesHistory': 10,
    'maxlenOfQueue': 200000,

    # MCTS
    'numMCTSSims': 64,    # 训练64–96均可；原论文围棋1600/棋类800，此处按算力下调。:contentReference[oaicite:0]{index=0}
    'cpuct': 1.5,         # 常用区间~1–2；文献/复现多用≈1.5。:contentReference[oaicite:1]{index=1}
    'tempThreshold': 10,  # 先高温后降0；围棋~前30手，这里棋盘更小取~10。:contentReference[oaicite:2]{index=2}

    # 评测/门控
    'arenaCompare': 64,   # 偶数且≥64便于换边统计；DeepMind评测很多局，门槛≈55%。:contentReference[oaicite:3]{index=3}
    'evaluate': 128,      # 确认赛更多局，降低方差。:contentReference[oaicite:4]{index=4}
    'updateThreshold': 0.55,  # 经典门槛≈55%。:contentReference[oaicite:5]{index=5}

    # 模型/存档
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': 'best.pth.tar',
    'experiment': 0,

    # 根注噪（只用于自博弈）
    'dirichlet_alpha': 0.15,  # 经验可按 α≈10/合法步数；Go用0.03(≈10/362)。:contentReference[oaicite:6]{index=6}
    'noise_eps': 0.1,        # 论文/实现通用值。:contentReference[oaicite:7]{index=7}
})
