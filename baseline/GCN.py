'''
Descripttion: 
version: 
Author: Rui Dong
Date: 2023-10-07 19:58:24
LastEditors: Please set LastEditors
LastEditTime: 2023-10-18 17:53:31
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

#* Just test the code
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class GCN(nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.1, nb_layers=3):
        super(GCN, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout

        self.conv1 = GCNConv(self.features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def forward(self, data, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)

        x = self.fc_forward(x)

        return x

    def __repr__(self):
        return self.__class__.__name__


"""
======================= Below are test for the demo ========================
"""

def train(model):
    model.train()
    
    for data in train_loader:
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()



def test(model, dataloader):
    model.eval()
    correct = 0
    for data in dataloader:                                 # 批遍历测试集数据集。
        out = model(data)    # 一次前向传播
        pred = out.argmax(dim=1)                            # 使用概率最高的类别
        correct += int((pred == data.y).sum())              # 检查真实标签
    return correct / len(dataloader.dataset)

if __name__ == '__main__':
    dataset = TUDataset('../dataset/TUDataset', name='MUTAG')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[ : 150 ]
    test_dataset = dataset[ 150 : ]
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_classes = dataset.num_classes
    in_size = dataset.num_node_features
    model = GCN(in_size=in_size, nb_class=num_classes, d_model=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 121):
        train(model)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')