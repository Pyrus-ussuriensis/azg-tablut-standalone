import os
import sys
import time

import numpy as np
from tqdm import tqdm

#sys.path.append('../../')
from tablut.utils.utils import *

from tablut.father_class.NeuralNet import NeuralNet
from tablut.models.RandomData import RandomSymDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .TaflNNet import TaflNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())
        # 推断棋盘边长 n（最后两维）
        b0 = examples[0][0]
        B0 = b0.astype(np.float32) if hasattr(b0, "astype") else np.array(b0, np.float32)
        n = B0.shape[-1]; assert B0.shape[-2] == n, "board must be square"

        perms = action_perms(n)
        dl = DataLoader(
            RandomSymDataset(examples, n, perms),
            batch_size=args.batch_size,
            shuffle=True,
            #num_workers=0,#getattr(args, "num_workers", 2),
            pin_memory=True,#getattr(args, "cuda", False),
            #persistent_workers=2,#getattr(args, "num_workers", 0) > 0,
            drop_last=True,
        )

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses, v_losses = AverageMeter(), AverageMeter()
            t = tqdm(dl, desc='Training Net')
            for boards, target_pis, target_vs in t:
                boards = boards.float()
                target_pis = target_pis.float()
                target_vs = target_vs.float()

                if args.cuda:
                    boards = boards.contiguous().cuda(non_blocking=True)
                    target_pis = target_pis.contiguous().cuda(non_blocking=True)
                    target_vs = target_vs.contiguous().cuda(non_blocking=True)

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v  = self.loss_v (target_vs,  out_v)
                loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(),  boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad(); loss.backward(); optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        img2d = np.array(board.getImage(), dtype=np.int8)
        img = getNNImage(img2d, board.size, board.time)

        x = torch.from_numpy(img).unsqueeze(0)  # (1,C,H,W)
        if args.cuda:
            x = x.contiguous().cuda()

        #board = torch.FloatTensor(board.astype(np.float64)) # 此处进行了格式转换从而能够输入网络。
        #img = img.view(6, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(x)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
