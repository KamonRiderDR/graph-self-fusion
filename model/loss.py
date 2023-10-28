'''
Description: 
Author: Rui Dong
Date: 2023-10-27 18:49:02
LastEditors: Please set LastEditors
LastEditTime: 2023-10-27 20:43:27
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO
'''
    classification loss + contrastive loss + combinatorial loss
'''
class TripletContrastiveLoss(nn.Module):
    def __init__(self):
        super(TripletContrastiveLoss, self).__init__()
    
    def forward(self, args, out_gcn, out_trans, out_fusion, labels):
        lam1 = args.lam1
        lam2 = args.lam2
        lam3 = 1.00 - lam1 - lam2
        loss_gcn = F.cross_entropy(out_gcn, labels.view(-1))
        loss_trans = F.cross_entropy(out_trans, labels.view(-1))
        loss_fusion = F.cross_entropy(out_fusion, labels.view(-1))
        
        