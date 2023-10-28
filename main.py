'''
Description: BUG FROM HERE! (Maybe reconstruct later)
Author: Rui Dong
Date: 2023-10-25 20:28:11
LastEditors: Rui Dong
LastEditTime: 2023-10-26 14:18:46
'''

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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


parser = argparse.ArgumentParser(description="uni-graph with multimodal self fusion")
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
#* dataset config
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
parser.add_argument("--num_features", type=int)
parser.add_argument("--num_heads", type=int, default=8, help="number of heads of graphTrans")
parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
parser.add_argument("--pos_encoding", choices=[None, 'gcn', 'gin'], default='gcn')
parser.add_argument('--att_dropout', type=float, default=0.1, help='multi-head attention dropout ratio')
parser.add_argument('--d_k', type=int, default=64, help='dim of key matrix')
parser.add_argument('--d_v', type=int, default=64, help='dim of value matrix')
parser.add_argument('--pos_embed_type', type=str, default="s", help="type of positional embedding(s -> start m-> midd)")
# mixup layer parameters
parser.add_argument("--alpha", type=float, default=0.5, help="mix-up ratio")
parser.add_argument("--num_fusion_layers", type=int, default=4, help="layers of the mix-up encoder")

#* training config
parser.add_argument("--loss_log", type=int, default=0, help="loss log ID")

args = parser.parse_args()

'''
TRAIN AND TEST FUNCTIONS.
'''

def train(model, train_loader):
    model.train()
    train_loss = 0.00
    for data in train_loader:
        data.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        # print(out.size())
        # print(data.y.size())
        loss = criterion(out, data.y)
        # loss = F.cross_entropy(out,data.y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(train_loader)


def test(model, test_loader):
    model.eval()
    correct = 0
    for data in test_loader:     
        data.to(args.device)                                # 批遍历测试集数据集
        out = model(data)                                   # 一次前向传播
        pred = out.argmax(dim=1)                            # 使用概率最高的类别
        correct += int((pred == data.y).sum())              # 检查真实标签
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    dataset = TUDataset('dataset/TUDataset', name='PROTEINS')
    torch.manual_seed(777)
    dataset = dataset.shuffle()
    train_size = int( 0.8 * len(dataset) )
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_classes = dataset.num_classes
    in_size = dataset.num_features
    args.in_size = in_size
    args.num_classes = num_classes
    args.input_node_dim = dataset.num_node_features
    args.input_edge_dim = dataset.num_edge_features
    args.output_dim = num_classes
    args.num_features = dataset.num_features
    
    print(args)

    # model = MultiScaleGCN(args)
    # model = GraphormerEncoder(args)
    # model = GraphTransformer(args)
    model = GraphMixupFusion(args)

    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_function = nn.L1Loss(reduction="sum")
    # loss = F.cross_entropy()
    best_test_acc = 0.00
    with open("/home/dongrui/code/graph_self_fusion/logs/loss_{}.txt".format(args.loss_log), "w+") as file:
        for epoch in range(0, 501):
            train_loss = train(model, train_loader)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            best_test_acc = max(best_test_acc, test_acc)
            file.write(f'Epoch: {epoch:03d},  Train loss: {train_loss:.6f},  Train Acc: {train_acc:.6f},  Test Acc: {test_acc:.6f}\n')
            print(f'Epoch: {epoch:03d},  Train loss: {train_loss:.6f},  Train Acc: {train_acc:.6f},  Test Acc: {test_acc:.6f}')
        file.write("{}\n".format(best_test_acc))
    
    file.close()