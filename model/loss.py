'''
Description: 
Author: Rui Dong
Date: 2023-10-27 18:49:02
LastEditors: Rui Dong
LastEditTime: 2023-11-01 10:44:35
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
        # is good enough?
        loss_classification = loss_gcn * lam1 + loss_trans * lam2 + loss_fusion * lam3
        
        z_gcn = F.normalize(out_gcn, dim=1)
        z_trans = F.normalize(out_trans, dim=1)
        z_fusion = F.normalize(out_fusion, dim=1)
        
        loss2 = torch.nn.MSELoss()
        dist_gt = loss2(out_gcn, out_trans)
        dist_ft = loss2(out_fusion, out_trans)
        dist_fg = loss2(out_fusion, out_gcn)
        theta1 = args.theta1    # gcn   <-> trans
        theta2 = args.theta2    # trans <-> fusion
        theta3 = args.theta3    # gcn   <-> fusion
        loss_contrastive = dist_fg * theta3 + dist_ft * theta2 - dist_gt * theta1
        
        return loss_contrastive + loss_classification
        