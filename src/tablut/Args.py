from tablut.utils.utils import *
args0 = dotdict({
    'numIters': 1000,
    'numEps': 100,
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

})

args = dotdict({
    'numIters': 60,
    'numEps': 200,
    'tempThreshold': 10,
    'updateThreshold': 0.58,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 200,
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
