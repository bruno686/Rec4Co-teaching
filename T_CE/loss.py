import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update

def loss_function_co_teaching(y1, y2, t, drop_rate):
    loss1 = F.binary_cross_entropy_with_logits(y1, t, reduce = False)
    loss2 = F.binary_cross_entropy_with_logits(y2, t, reduce = False)
    
    loss_mul1 = loss1 * t
    ind_sorted1 = np.argsort(loss_mul1.cpu().data).cuda()
    loss_sorted1 = loss1[ind_sorted1]
    
    loss_mul2 = loss2 * t
    ind_sorted2 = np.argsort(loss_mul2.cpu().data).cuda()
    loss_sorted2 = loss2[ind_sorted2]
    
    remember_rate = 1 - drop_rate
    num_remember1 = int(remember_rate * len(loss_sorted1))
    num_remember2 = int(remember_rate * len(loss_sorted2))	

    ind_update1 = ind_sorted1[:num_remember1]
    ind_update2 = ind_sorted2[:num_remember2]

    loss_update1 = F.binary_cross_entropy_with_logits(y1[ind_update2], t[ind_update2])
    loss_update2 = F.binary_cross_entropy_with_logits(y2[ind_update1], t[ind_update1])
    
    # loss_update1 = F.binary_cross_entropy_with_logits(y1[ind_update1], t[ind_update1])
    # loss_update2 = F.binary_cross_entropy_with_logits(y2[ind_update2], t[ind_update2])
    return loss_update1, loss_update2