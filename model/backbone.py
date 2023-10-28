'''
Description: This .py file defines backbone of the model.
Including:  TODO
        1.  two encoders for gcn and transformers seperately.
        2.  Fusion layer
        
Author: Rui Dong
Date: 2023-10-09 12:39:58
LastEditors: Rui Dong
LastEditTime: 2023-10-18 21:21:04
'''


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv

class GCN(nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
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


class GraphSAGE(nn.Module):
    def __init__(self, in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
        super(GraphSAGE, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout

        self.conv1 = SAGEConv(self.features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))

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


'''
    SGCN model uses multiple hop adj to get multi-scale GCN representations.
'''
class SGCN(nn.Module):
    def __init__(self, in_size, nb_class, d_model, k=2, dropout=0.1, nb_layers=4):
        super(SGCN, self).__init__()
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout
        self.k = k

        self.conv1 = SGConv(self.features, self.hidden_dim, self.k)
        # self.conv1 = GCNConv(self.features, self.hidden_dim)
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