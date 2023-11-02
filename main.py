'''
Description: BUG FROM HERE! (Maybe reconstruct later)
Author: Rui Dong
Date: 2023-10-25 20:28:11
LastEditors: Rui Dong
LastEditTime: 2023-11-02 18:03:53
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
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

par_dir = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(par_dir)
sys.path.append(par_dir)
sys.path.append(os.path.join(par_dir, "utils"))
sys.path.append(os.path.split(sys.path[0])[0])

from model.backbone import GCN, SGCN
from model.layers import MultiScaleGCN, GraphormerEncoder, GraphTransformer, GraphMixupFusion
from model.graph_self_fusion import GraphSelfFusion
from model.loss import TripletContrastiveLoss 
from utils.utils import k_fold


config_dir = "/home/dongrui/config/"

parser = argparse.ArgumentParser(description="uni-graph with multimodal self fusion")
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
#* dataset config
parser.add_argument('--dataset', type=str, default="PROTEINS", help="TUDataset: MUTAG, PROTEINS, NCI1...")
parser.add_argument("--in_size", type=int, help="input size of graph features")
parser.add_argument("--num_classes", type=int, help="number of classes of the graph")
#* model config
#  GCN parameters
parser.add_argument("--fusion_type", type=str, default="early", help="GCN fusion strategy")
parser.add_argument("--gcn_channels", type=int, default=3, help="number of channels of the GCN encoders")
parser.add_argument("--gcn_hidden", type=int, default=64, help="hidden dim of gcn encoder")
parser.add_argument("--gcn_layers", type=int, default=4, help="number of GCN encoder layers")
parser.add_argument("--gcn_dropout", type=float, default=0.1, help="dropout of GCN encoder")
#  graphormer parameters
parser.add_argument("--trans_num_layers", type=int, default=4, help="number of graphormer encoder layers")
parser.add_argument("--input_node_dim", type=int, help="input size of node features")
parser.add_argument("--hidden_node_dim", type=int, default=32, help="hidden dim of graphormer encoder")
parser.add_argument("--input_edge_dim", type=int, help="input size of edge features")
parser.add_argument("--hidden_edge_dim", type=int, default=32, help="hidden dim of graphormer encoder")
parser.add_argument("--ouput_dim", type=int, help="ouput dim of graphormer encoder")
parser.add_argument("--n_heads", type=int, default=4, help="number of heads of graphormer encoder")
parser.add_argument("--max_in_degree", type=int, default=5, help="max in degree")
parser.add_argument("--max_out_degree", type=int, default=5, help="max out degree")
parser.add_argument("--max_path_distance", type=int, default=5, help="max path distance")
#  graphTrans parameters
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dim of graphTrans")
parser.add_argument("--num_layers", type=int, default=4, help="number of graphTrans layers")
parser.add_argument("--num_features", type=int) # same as in_size
parser.add_argument("--num_heads", type=int, default=8, help="number of heads of graphTrans")
parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
parser.add_argument("--pos_encoding", choices=[None, 'gcn', 'gin'], default='gcn')
parser.add_argument('--att_dropout', type=float, default=0.1, help='multi-head attention dropout ratio')
parser.add_argument('--d_k', type=int, default=64, help='dim of key matrix')
parser.add_argument('--d_v', type=int, default=64, help='dim of value matrix')
parser.add_argument('--pos_embed_type', type=str, default="s", help="type of positional embedding(s -> start m-> midd)")
# mixup layer && fusion model parameters
parser.add_argument("--alpha", type=float, default=0.5, help="mix-up ratio")
parser.add_argument("--num_fusion_layers", type=int, default=4, help="layers of the mix-up encoder")
parser.add_argument("--eta", type=float, default=0.4, help="fusion pattern mix-ratio with residual pattern")
# loss parameters
parser.add_argument("--lam1", type=float, default=0.2, help="lam1 is for loss_gcn")
parser.add_argument("--lam2", type=float, default=0.2, help="lam2 is for loss_trans")
parser.add_argument("--theta1", type=float, default=0.1, help="theta1 is for loss_l2_gt")
parser.add_argument("--theta2", type=float, default=0.4, help="theta2 is for loss_l2_ft")
parser.add_argument("--theta3", type=float, default=0.4, help="theta3 is for loss_l2_fg")

#* training config
parser.add_argument("--loss_log", type=int, default=0, help="loss log ID")
parser.add_argument('--folds', type=int, default=10, help='number of k-folds (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epoches', type=int, default=500, help='maximum number of epochs')


args = parser.parse_args()

'''
TRAIN AND TEST FUNCTIONS.
'''

def train_epoch(args, model, optimizer, train_loader):
    # criterion = torch.nn.CrossEntropyLoss() # define loss function
    criterion = TripletContrastiveLoss()
    model.train()
    train_loss = 0.00
    for data in train_loader:
        data.to(args.device)
        optimizer.zero_grad()
        out_gcn, out_trans, out = model(data)
        loss = criterion(args, out_gcn, out_trans, out, data.y)
        # loss = criterion(out, data.y)
        # loss = F.cross_entropy(out,data.y)
        loss.backward()
        args.optimizer.step()
        
        train_loss += loss.item()
    # return train_loss / len(train_loader)

# also for eval_epoch
'''
description: 
param {*} args
param {*} model
param {*} test_loader
return {*}      ACC && LOSS
'''
def test_epoch(args, model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0.
    criterion = TripletContrastiveLoss()
    for data in test_loader:     
        data.to(args.device)                                # 批遍历测试集数据集
        out_gcn, out_trans, out = model(data)               # 一次前向传播
        pred = out.argmax(dim=1)                            # 使用概率最高的类别
        correct += int((pred == data.y).sum())              # 检查真实标签
        test_loss += criterion(args, out_gcn, out_trans, out, data.y).item() * out.shape[0]
    return correct / len(test_loader.dataset), test_loss / len(test_loader.dataset)


'''
@description:   train * epoches times [in one fold]
@param: 
@return:        best val-acc, best test-acc
'''
def train_model(args, model, optimizer, 
                train_loader, val_loader, test_loader,
                i_fold):
    min_loss = 1e10
    max_acc = 0.00
    best_epoch = 0
    # criterion = torch.nn.CrossEntropyLoss() # define loss function
    criterion = TripletContrastiveLoss()
    test_accs = []
    
    for epoch in range(args.epoches):
        model.train()
        train_loss = 0.00
        for data in train_loader:
            data.to(args.device)
            optimizer.zero_grad()
            out_gcn, out_trans, output =  model(data)
            
            loss = criterion(args, out_gcn, out_trans, output, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        val_acc, val_loss = test_epoch(args, model, val_loader)
        test_acc, test_loss = test_epoch(args, model, test_loader)
        test_accs.append(test_acc)
        print('Epoch: {:03d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
                'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
                'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc))
        #   验证集效果最好的用在测试集上
        if val_acc > max_acc:
            max_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_weights)
    test_acc, test_loss_ = test_epoch(args, model, test_loader)
    return max_acc, test_acc

def k_fold_train(args, model, dataset, folds):
    val_accs = []
    test_accs = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        print('Fold: {:1d}'.format(fold))
        model.reset_parameters()
        #* load in model TODO
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        
        max_acc, test_acc = train_model(args, model, optimizer, train_loader, val_loader, test_loader, fold)
        val_accs.append(max_acc)
        test_accs.append(test_acc)
    
    print(test_accs)
    print(            
        "[{:d} Fold results] data_val:{:.2f} ± {:.2f} data_test:{:.2f} ± {:.2f}".format(
        folds, 
        np.mean(val_accs) * 100, np.std(val_accs) * 100,
        np.mean(test_accs) * 100, np.std(test_accs) * 100)
        )
    
        


if __name__ == '__main__':
    dataset = TUDataset('dataset/TUDataset', name=args.dataset)
    torch.manual_seed(777)
    dataset = dataset.shuffle()
    train_size = int( 0.8 * len(dataset) )
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_classes = dataset.num_classes
    in_size = dataset.num_features
    args.in_size = in_size
    args.num_classes = num_classes
    args.input_node_dim = dataset.num_node_features
    args.input_edge_dim = dataset.num_edge_features
    args.output_dim = num_classes
    args.num_features = dataset.num_features    
    print(args)
    
    # model = GraphMixupFusion(args)
    model = GraphSelfFusion(args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    
    k_fold_train(args, model, dataset, folds=args.folds)