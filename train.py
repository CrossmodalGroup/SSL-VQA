import os
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import json
from torch.optim.lr_scheduler import MultiStepLR


# standard cross-entropy loss
def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg,dim=-1), 1, top_ans_ind).sum(1)
    
    qice_loss = neg_top_k.mean()
    return qice_loss


def train(model, train_loader, eval_loader, opt):

    utils.create_dir(opt.output)
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=opt.weight_decay)
    logger = utils.Logger(os.path.join(opt.output, 'log.txt'))

    utils.print_model(model, logger)

    # load snapshot
    if opt.checkpoint_path is not None:
        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

    for param_group in optim.param_groups:
        param_group['lr'] = opt.learning_rate

    scheduler = MultiStepLR(optim, milestones=[10,15,20,25,30,35], gamma=0.5)
    scheduler.last_epoch = opt.s_epoch

    

    best_eval_score = 0
    for epoch in range(opt.s_epoch, opt.num_epochs):
        total_loss = 0
        total_bce_loss = 0
        self_loss = 0
        total_self_loss = 0
        train_score_pos = 0
        train_score_neg = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        scheduler.step()

        for i, (v, b, q, a,_) in enumerate(train_loader):
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()

            # for the labeled samples
            if epoch < opt.pretrain_epoches:
                logits_pos, _= model(q, v, False)
                if opt.ml_loss:
                    bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                else:
                    bce_loss_pos = instance_bce(logits_pos, a)
                loss = bce_loss_pos
            else:
                logits_pos, logits_neg, _, _ = model(q, v, True)
                if opt.ml_loss: #use multi-label loss
                    bce_loss_pos = instance_bce_with_logits(logits_pos, a, reduction='mean')
                else: #use cross-entropy loss
                    bce_loss_pos = instance_bce(logits_pos, a)
            
                self_loss = compute_self_loss(logits_neg, a)

                loss = bce_loss_pos + opt.self_loss_weight * self_loss

            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            score_pos = compute_score_with_logits(logits_pos, a.data).sum()
            train_score_pos += score_pos.item()
            total_loss += loss.item() * v.size(0)
            total_bce_loss += bce_loss_pos.item() * v.size(0)

            if epoch < opt.pretrain_epoches: #pretrain
                total_self_loss = 0
                train_score_neg = 0
            else: #fintune
                score_neg = compute_score_with_logits(logits_neg, a.data).sum()
                total_self_loss += self_loss.item() * v.size(0)
                train_score_neg += score_neg.item()
            if i != 0 and i % 100 == 0:
                print(
                    'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, self_loss: %.6f, neg_train_acc: %.6f, pos_train_acc: %.6f' %
                    (i, len(train_loader), total_loss / (i * v.size(0)),
                     total_bce_loss / (i * v.size(0)),
                     total_self_loss / (i * v.size(0)),
                     100 * train_score_neg / (i * v.size(0)), 100 * train_score_pos / (i * v.size(0))))

        total_loss /= N
        total_bce_loss /= N
        total_self_loss /= N
        train_score_pos = 100 * train_score_pos / N
        if None != eval_loader:
            model.train(False)
            eval_score, bound, entropy = evaluate(model, eval_loader)
            model.train(True)

        logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write(
            '\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm / count_norm, train_score_pos))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_loader is not None and entropy is not None:
            info = '' + ' %.2f' % entropy
            logger.write('\tentropy: ' + info)

        if (eval_loader is not None and eval_score > best_eval_score):
            model_path = os.path.join(opt.output, 'best_model.pth')
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = 0
    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        q_id = q_id.cuda()
        pred, att = model(q, v, False)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        entropy += calc_entropy(att.data)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)

    return score, upper_bound, entropy


def calc_entropy(att):  # size(att) = [b x v x q]
    sizes = att.size()
    eps = 1e-8
    # att = att.unsqueeze(-1)
    p = att.view(-1, sizes[1] * sizes[2])
    return (-p * (p + eps).log()).sum(1).sum(0)  # g

