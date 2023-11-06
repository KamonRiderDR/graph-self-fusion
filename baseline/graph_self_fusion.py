'''
Description:  This is the interface for final model
Author: Rui Dong
Date: 2023-10-27 09:46:47
LastEditors: Rui Dong
LastEditTime: 2023-11-06 21:15:59
'''

import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj


from .layers import MultiScaleGCNLayer, SelfMixupFusionLayer, SelfMultiFusionLayer, SelfFusionTransformerLayer
from .graphormer_layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding
from .graphTrans_layers import TransformerEncoder, PositionEncoder
from .functional import shortest_path_distance, batched_shortest_path_distance


class GraphSelfFusion(nn.Module):
    def __init__(self, args):
        super(GraphSelfFusion, self).__init__()
        self.args = args
        #* init parameters
        self.num_layers = args.num_fusion_layers
        # gcn layer parameters
        self.fusion_type    = args.fusion_type
        self.gcn_channels   = args.gcn_channels
        self.num_features   = args.in_size
        self.hidden_dim     = args.gcn_hidden
        self.num_layers     = args.num_fusion_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.gcn_dropout
        self.device         = args.device
        # trans layer parameters
        self.pos_encoding   = args.pos_encoding
        self.pos_embedding  = PositionEncoder(self.args,
                                            self.hidden_dim,
                                            pos_enc=self.pos_encoding)
        self.hidden_pos_embedding = PositionEncoder(self.args,
                                                    self.hidden_dim,
                                                    pos_enc=self.pos_encoding,
                                                    embedding_type="m")
        self.alpha          = args.alpha        # alpha is the ratio between gcn and transformer
        self.eta            = args.eta          # eta is fusion pattern mix parameters
        # fusion transformer parameters
        self.num_heads      = args.num_heads
        self.ffn_dim        = args.ffn_dim
        
        #* init encoder blocks
        self.gcn_encoder1 = MultiScaleGCNLayer(
                self.gcn_channels,
                self.num_features,
                self.hidden_dim,
                self.hidden_dim,
                self.dropout,
                self.device
        )
        self.trans_encoder1 = TransformerEncoder(self.args)
        self.encoders = torch.nn.ModuleList()
        self.fusion_transformer_layers = nn.ModuleList()
        for i in range(self.num_layers-1):
            alpha_1 = np.random.normal(loc=self.alpha, scale=self.alpha)
            if alpha_1 >= 1:
                alpha_1 = self.alpha
            self.encoders.append(SelfMultiFusionLayer(self.args, position="m"))
            self.encoders[i].reset_alpha(alpha_1)
            self.fusion_transformer_layers.append(SelfFusionTransformerLayer(
                self.hidden_dim, self.num_heads, self.ffn_dim, self.dropout 
            ))

        '''Define MLP for final output head'''
        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                # torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, hidden // 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden // 2, outchannel),
            )

        self.fc_gcn     = mlp(self.hidden_dim, self.hidden_dim, self.num_classes)
        self.fc_trans   = mlp(self.hidden_dim, self.hidden_dim, self.num_classes)
        self.fc_fusion  = mlp(self.hidden_dim, self.hidden_dim, self.num_classes)
        
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)

    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        self.hidden_pos_embedding.reset_parameters()
        self.gcn_encoder1.reset_parameters()
        self.trans_encoder1.reset_parameters()
        for encoder in self.encoders:
            encoder.reset_parameters()
        # Hard coding
        self.fc_gcn[0].reset_parameters()
        self.fc_trans[0].reset_parameters()
        self.fc_fusion[0].reset_parameters()
        self.fc_gcn[2].reset_parameters()
        self.fc_trans[2].reset_parameters()
        self.fc_fusion[2].reset_parameters()
        self.fc_gcn[4].reset_parameters()
        self.fc_trans[4].reset_parameters()
        self.fc_fusion[4].reset_parameters()


    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x
    
    
    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        x_ = self.pos_embedding(x, edge_index)
        batch_x, mask = to_dense_batch(x_, batch)
        adj = to_dense_adj(edge_index, batch)
        # here, tensor.shape 2 -> 3
        x_gcn = self.gcn_encoder1(x, edge_index)
        x_trans = self.trans_encoder1(batch_x, mask=mask, att_bias=adj)
        x_trans = x_trans[mask]
        x_gcn_mix = x_gcn * self.alpha
        x_trans_mix = x_trans * (1.00 - self.alpha)
        # 3* input and 3* output
        x_mix = torch.add(x_gcn_mix, x_trans_mix)

        for encoder in self.encoders:
            res_mix = x_mix
            x_gcn, x_trans, x_mix = encoder(x_gcn, x_trans, x_mix, edge_index, batch) # TODO
            x_mix = torch.add(x_mix * self.eta,  res_mix * (1.00-self.eta) ) 
        
        # output head
        x_gcn = global_add_pool(x_gcn, batch)
        x_trans = global_add_pool(x_trans, batch)
        x_mix = global_add_pool(x_mix, batch)
        x_gcn = self.fc_gcn(x_gcn)
        x_trans = self.fc_trans(x_trans)
        x_mix = self.fc_fusion(x_mix)

        return x_gcn, x_trans, x_mix
