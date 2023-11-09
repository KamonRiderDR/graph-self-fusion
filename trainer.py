'''
Description: 
Author: Rui Dong
Date: 2023-11-06 11:16:40
LastEditors: Please set LastEditors
LastEditTime: 2023-11-09 16:56:22
'''

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
# from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from model.backbone import GCN, SGCN
from model.layers import MultiScaleGCN, GraphormerEncoder, GraphTransformer, GraphMixupFusion
from model.graph_self_fusion import GraphSelfFusion, GraphSelfFusionMix, GraphSelfFusionTransMix
from model.loss import TripletContrastiveLoss 
from utils.utils import k_fold
from utils.dataset import TUDataset # Only for IMDB-only


def generate_model(args):
    if args.model_name == "fusion":
        return GraphSelfFusion(args)
    elif args.model_name == "fusion_mix":
        return GraphSelfFusionMix(args)
    elif args.model_name == "fusion_tm":
        return GraphSelfFusionTransMix(args)

class Trainer:
    """ Description: Trainer is used to execute one training  && evaluating && testing
    process. Include k fold train, eval and train epoch.
    """
    def __init__(self, args):
        # self.args = args
        self.model = None
        self.scheduler = None
        self.optimizer = None
        # self.criterion = TripletContrastiveLoss()
        
    
    def k_fold_train(self, args, dataset, folds):
        val_accs = []
        best_tests = []
        test_accs = []
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
            print('Fold: {:1d}'.format(fold))
            #* model reset
            self.model = generate_model(args)
            self.model.to(args.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.8)

            #* load in dataset and dataloaders
            train_dataset = dataset[train_idx]
            val_dataset = dataset[val_idx]
            test_dataset = dataset[test_idx]
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

            #* train && eval
            best_val_acc, max_test_acc, test_acc = self.train_epoch(args, train_loader, val_loader, test_loader)
            val_accs.append(best_val_acc)
            test_accs.append(test_acc)
            best_tests.append(max_test_acc)
            print("------------- {:d} val_acc: {:.4f} test_acc: {:.4f} best_test: {:.4f} -----------------".format(
                fold, 
                best_val_acc, test_acc, max_test_acc
            ))
            torch.cuda.empty_cache()

        #* record
        print(test_accs)
        print(best_tests)
        print(            
            "[{:d} Fold results] data_val:{:.2f} ± {:.2f} data_test:{:.2f} ± {:.2f} best_test:{:.2f} ± {:.2f}".format(
            folds, 
            np.mean(val_accs) * 100, np.std(val_accs) * 100,
            np.mean(test_accs) * 100, np.std(test_accs) * 100,
            np.mean(best_tests) * 100, np.std(best_tests) * 100)
            )
    
    
    def train_epoch(self, args, train_loader, val_loader, test_loader):
        '''
        @description:   train * epoches times [in one fold]
        @param: 
        @return:        best val-acc, best test-acc(during training process), best test-acc
        '''        
        best_val_acc = 0.00
        best_test_acc = 0.00
        test_accs = []
        criterion = TripletContrastiveLoss()
        patience = 0
        
        for epoch in range(args.epoches):
            self.model.train()
            train_loss = 0.00
            for data in train_loader:
                data.to(args.device)
                self.optimizer.zero_grad()
                out_gcn, out_trans, output =  self.model(data)
                
                loss = criterion(args, out_gcn, out_trans, output, data.y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            self.scheduler.step()
            
            train_loss = train_loss / len(train_loader.dataset)
            val_acc, val_loss = self.test_epoch(args, val_loader)
            test_acc, test_loss = self.test_epoch(args, test_loader)
            test_accs.append(test_acc)
            # best_test_acc = max(best_test_acc, test_acc)

            if epoch % 10 == 0:
                print('Epoch: {:03d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
                    'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
                    'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc))
            if test_acc > best_test_acc:
                # best_test_acc = test_acc
                best_test_weights = copy.deepcopy(self.model.state_dict())
            #   验证集效果最好的用在测试集上
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(self.model.state_dict())
                patience = 0
            elif val_acc == best_val_acc and test_acc > best_test_acc: 
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_weights = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
            
            if patience == args.patience:
                print("Early stop at epoch {:03d}.".format(epoch))
                break
        
        self.model.load_state_dict(best_weights)
        test_acc, test_loss_ = self.test_epoch(args, test_loader)
        self.model.load_state_dict(best_test_weights)
        best_test_acc, best_test_loss = self.test_epoch(args, test_loader)
        return best_val_acc, best_test_acc, test_acc
            
            
            
    def test_epoch(self, args, test_loader):
        '''
        description: Test, also for eval_epoch.
        param {*} args
        param {*} model
        param {*} test_loader
        return {*}      acc && loss
        '''
        self.model.eval()
        correct = 0
        test_loss = 0.
        criterion = TripletContrastiveLoss()
        for data in test_loader:     
            data.to(args.device)                                     # 批遍历测试集数据集
            out_gcn, out_trans, out = self.model(data)               # 一次前向传播
            pred = out.argmax(dim=1)                                 # 使用概率最高的类别
            correct += int((pred == data.y).sum())                   # 检查真实标签
            test_loss += criterion(args, out_gcn, out_trans, out, data.y).item() * out.shape[0]
        return correct / len(test_loader.dataset), test_loss / len(test_loader.dataset)
    
    
    '''
    description:    Test all datasets * k_fold_train 
    param {*} self
    param {*} args
    param {*} folds
    param {*} dataset_path      TUDataset path
    param {*} dataset_list      TUDataset name list
    return {*}
    '''    
    def train_all_datasets(self, args, folds, dataset_path, dataset_list):
        pass