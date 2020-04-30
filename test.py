"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_vqacp import Dictionary, VQAFeatureDataset
from base_model import Model
import utils
import opts



def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        logits, att = model(q,v,False)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device


    model = Model(opt)
    model = model.cuda()

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    def process(args, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

        model.train(False)

        logits, qIds = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
        model_label = opt.label 
        
        if opt.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % opt.s_epoch)
        
        utils.create_dir(opt.output)
        if 0 <= opt.s_epoch:
            model_label += '_epoch%d' % opt.s_epoch

        with open(opt.output+'/test_%s.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)
