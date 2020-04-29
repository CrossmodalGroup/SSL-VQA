import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from dataset_vqacp import Dictionary, VQAFeatureDataset

from base_model import Model
import utils
import opts
from train import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    model = Model(opt)
    model = model.cuda()
    model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)

    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils.trim_collate)
    opt.use_all = 1
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils.trim_collate)

    train(model, train_loader, eval_loader, opt)
