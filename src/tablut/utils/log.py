import logging
import os, torch
from logging.handlers import RotatingFileHandler
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter
from tablut.Args import *

def init_logging():
    # 建立Tensorboard的写对象
    if args.load_model:
        meta = torch.load(os.path.join(args.checkpoint, "resume.pt"), map_location="cpu")
        writer = SummaryWriter(log_dir=meta["writer_path"], purge_step=meta["i"])
    else:
        writer = SummaryWriter(log_dir='tensorboard/'+f'experiment{args.experiment}')
    # 建立日志记录对象
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 文件最大 10MB，保留 5 个备份
    handler = RotatingFileHandler(f'logs/experiment{args.experiment}.log', maxBytes=10*1024*1024, backupCount=5)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    # 记录参数的配置
    logger.info("current configuration\n%s", pformat(args))
    logger.info('start training')
    return logger, writer

logger, writer = init_logging()

'''
# 训练和验证时日志和Tensorboard的统一记录
def log_info(epoch, loss, mode, place):
    info = f"mode: {mode}\n{place}: {epoch}\nloss: {loss}\n\n"
    #print(info)
    logger.info(info)
    if mode == "train":
        writer.add_scalar(f'{place}/train/loss', loss, epoch)
        #writer.add_scalar('train/loss', loss, epoch)
    elif mode == "val":
        writer.add_scalar(f'{place}/val/loss', loss, epoch)
        #writer.add_scalar('val/loss', loss, epoch)
    else:
        error_info = "writer mode error!!!"
        #print(error_info)
        logger.error(error_info)
    writer.flush()
'''     




