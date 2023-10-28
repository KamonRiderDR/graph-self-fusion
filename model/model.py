import numpy as np
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data

from .layers import MultiScaleLayer, MultiScaleGCNLayer, SelfMixupFusionLayer
from .graphormer_layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding
from .functional import shortest_path_distance, batched_shortest_path_distance



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.scale = (args.r - 1) * args.Lev + 1

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.multi_scale = MultiScaleLayer(args).to(args.device)

        self.fc1 = nn.Linear(self.scale * self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)
        return x

    def forward(self, data, mixup=False, alpha=0.1):
        x, y, edge_index, batch, d, d_index = data.x, data.y, data.edge_index, data.batch, data.d, data.d_index
        batch_size = int(batch.max() + 1)

        if mixup:
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            perm = torch.randperm(y.size(0), device=self.args.device)
            y_perm = y[perm]

            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

            x = self.multi_scale(x, batch, batch_size, d, d_index)
            x_mix = lam * x + (1 - lam) * x[perm, :]

            x = self.fc_forward(x_mix)

            return x, y_perm, lam
        else:
            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = self.multi_scale(x, batch, batch_size, d, d_index)

            x = self.fc_forward(x)

            return x


class ModelHierarchical(nn.Module):
    def __init__(self, args):
        super(ModelHierarchical, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.scale = (args.r - 1) * args.Lev + 1

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.multi_scale = MultiScaleLayer(args).to(args.device)

        self.fc1 = nn.Linear(self.scale * self.hidden_dim * self.num_layers, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)
        return x

    def forward(self, data, mixup=False, alpha=0.1):
        x, y, edge_index, batch, d, d_index = data.x, data.y, data.edge_index, data.batch, data.d, data.d_index
        batch_size = int(batch.max() + 1)

        if mixup:
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            perm = torch.randperm(y.size(0), device=self.args.device)
            y_perm = y[perm]

            x = F.relu(self.conv1(x, edge_index))
            x_h = self.multi_scale(x, batch, batch_size, d, d_index)
            x_mix = lam * x_h + (1 - lam) * x_h[perm, :]
            xs = [x_mix]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x_h = self.multi_scale(x, batch, batch_size, d, d_index)
                x_mix = lam * x_h + (1 - lam) * x_h[perm, :]
                xs += [x_mix]
            x = torch.cat(xs, dim=-1)
            x = self.fc_forward(x)

            return x, y_perm, lam
        else:
            x = F.relu(self.conv1(x, edge_index))
            x_h = self.multi_scale(x, batch, batch_size, d, d_index)
            xs = [x_h]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x_h = self.multi_scale(x, batch, batch_size, d, d_index)
                xs += [x_h]
            x = torch.cat(xs, dim=-1)
            x = self.fc_forward(x)

            return x
        
        
        
class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x
