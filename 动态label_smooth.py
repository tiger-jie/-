import torch
import torch.nn.functional as F


def cal_loss(pred, gold, mask=None, smooth=False, eps=None):
    if smooth:
        gold = gold.view(-1, 1)
        n_class = mask.sum(1).view(-1, 1)
        one_hot = torch.zeros_like(pred).scatter(1, gold, 1)
        smooth_label = one_hot * (1.0 - eps) + (1.0 - one_hot) * eps / (n_class-1)
        pred = pred.masked_fill(mask == 0, 1e-9)
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(log_prob*smooth_label)
        loss = loss.masked_fill(mask == 0, 0)
        loss = loss.sum(dim=1).mean()
        return loss
    else:
        return F.cross_entropy(pred, gold)




