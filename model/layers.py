'''
Description: This .py file is used to define basic layer of the graph model. 
INCLUDING: 
        1.  Multi-hop GCN layers
        2.	Mixup-fusion for two graph representations
        3. 	Multi-modal fusion layer
Author: Rui Dong
Date: 2023-10-10 08:55:26
LastEditors: Please set LastEditors
LastEditTime: 2023-10-27 20:43:47
'''

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv
from torch_geometric.data import Data
from torch_sparse import spmm
from torch_geometric.utils import to_dense_batch, to_dense_adj

from .backbone import GCNConv, GraphSAGE, SGCN
from .graphormer_layers import GraphormerEncoderLayer,  CentralityEncoding, SpatialEncoding
from .graphTrans_layers import TransformerEncoder, PositionEncoder
from .functional import shortest_path_distance, batched_shortest_path_distance

# 相对导入有问题
from utils import *


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = args.num_heads
        self.dropout = args.att_dropout
        self.d_model = args.hidden_dim
        self.d_k = args.d_k
        self.d_v = args.d_v

        self.W_Q = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_model, self.num_heads * self.d_v, bias=False)

        self.fc = nn.Linear(self.num_heads * self.d_v, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def reset_parameters(self):
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, mask=None, att_bias=None):
        residual = x
        x = self.layer_norm(x)

        batch_size = x.size(0)
        x_dim = x.size(1)

        Q = self.W_Q(x).view(batch_size, x_dim, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, x_dim, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, x_dim, self.num_heads, self.d_v).transpose(1, 2)

        att_weights = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(self.d_model)

        if mask is not None:
            att_weights = att_weights.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if att_bias is not None:
            att_bias = torch.repeat_interleave(att_bias, repeats=self.num_heads, dim=0).view(batch_size, self.num_heads, x_dim, x_dim)
            att_weights = att_weights + att_bias

        att_weights = torch.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)

        attention = torch.matmul(att_weights, V)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, x_dim, -1)
        output = self.fc(attention)
        output += residual

        return output


    
'''
    K-hop GCN layer for early fusion
'''    
class MultiScaleGCNLayer(nn.Module):
    def __init__(self, gcn_channels,
                        in_size,
                        hidden_dim,
                        out_dim,
                        dropout,
                        device='cuda:0'):
        super(MultiScaleGCNLayer, self).__init__()

        self.gcn_channels   = gcn_channels
        self.features       = in_size
        self.hidden_dim     = hidden_dim
        self.out_dim        = out_dim
        self.dropout        = dropout
        self.device         = device

        self.convs = torch.nn.ModuleList()
        self.convs2= torch.nn.ModuleList()
        self.convs3 = torch.nn.ModuleList()
        assert(self.gcn_channels>=1)
        for i in range(self.gcn_channels):
            conv_1      = SGConv(self.features, self.hidden_dim, i+1)
            conv_2      = SGConv(self.hidden_dim, self.hidden_dim, i+1) 
            conv_3      = SGConv(self.hidden_dim, self.hidden_dim, i+1)

            self.convs.append(conv_1)
            self.convs2.append(conv_2)
            self.convs3.append(conv_3)
        
        self.fc = torch.nn.Linear(self.hidden_dim, self.out_dim)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for conv in self.convs3:
            conv.reset_parameters()
        self.fc.reset_parameters()
    
    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
    #     X_k = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
    #     # for conv in self.convs:
    #     #     X_k = torch.add(X_k, conv(x, edge_index))
    #     for i in range(self.gcn_channels):
    #         X_k1 = self.convs[i](x, edge_index)
    #         X_k1 = self.convs2[i](X_k1, edge_index)
    #         X_k1 = self.convs3[i](X_k1, edge_index)
    #         X_k = torch.add(X_k, X_k1)
    #     X_k = torch.div(X_k, self.gcn_channels)
    #     return self.fc(X_k)
    def forward(self, x, edge_index):
        X_k = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        # for conv in self.convs:
        #     X_k = torch.add(X_k, conv(x, edge_index))
        for i in range(self.gcn_channels):
            X_k1 = self.convs[i](x, edge_index)
            X_k1 = self.convs2[i](X_k1, edge_index)
            X_k1 = self.convs3[i](X_k1, edge_index)
            X_k = torch.add(X_k, X_k1)
        X_k = torch.div(X_k, self.gcn_channels)
        return self.fc(X_k)




'''    
	@description:   Mixup Fusion Layer is for two multimodal graph layer. 
	The input is representation of two graphs.

    @return:        Fusion representation
'''
class SelfMixupFusionLayer(nn.Module):
    def __init__(self, args, position="s"):
        super(SelfMixupFusionLayer, self).__init__()
        #* gcn_channels is for multi-hop [1, gcn_channels] gcn kernels 
        self.args = args

        self.features       = args.in_size
        self.hidden_dim     = args.hidden_dim
        self.num_layers     = args.gcn_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.dropout
        self.alpha          = args.alpha
        self.device         = args.device
        self.pos_enc        = args.pos_encoding
        self.pos_embedding  = PositionEncoder(self.args, self.hidden_dim, pos_enc=self.pos_enc,
                                                embedding_type=position)
        
        #* GCN*1 + Trans*1
        # hidden -> hidden
        self.GCN_layer = MultiScaleGCNLayer(
            gcn_channels=args.gcn_channels,
            in_size=args.gcn_hidden,       # mid stage so the input is hidden_dim
            hidden_dim=args.gcn_hidden,
            out_dim=args.gcn_hidden,
            dropout=args.dropout,
            device=self.device
        )
        self.transformer_layer = TransformerEncoder(self.args)
        
        self.fc1 = nn.Linear(self.features, self.features)
        # self.gcns = torch.nn.ModuleList()
        # for i in range(self.gcn_channels):
        #     self.gcns.append(SGCN(self.features, 
        #                         self.num_classes, self.hidden_dim, 
        #                         i+1,
        #                         self.dropout, self.num_layers))    # TODO
    
    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        self.GCN_layer.reset_parameters()
        self.transformer_layer.reset_parameters()
        self.fc1.reset_parameters()
    
    def reset_alpha(self, alpha):
        assert alpha <= 1
        self.alpha = alpha
    
    # def graphTrans_fc(self, data):
    #     x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
    #     out = self.pos_embedding(x, edge_index)
    #     batch_x, mask = to_dense_batch(out, batch)
    #     adj = to_dense_adj(edge_index, batch)        
    #     out = self.transformer_layer(batch_x, mask=mask, att_bias=adj)
        
    #     return out
    def graphTrans_fc(self, x, edge_index, batch):
        out = self.pos_embedding(x, edge_index)
        batch_x, mask = to_dense_batch(out, batch)
        adj = to_dense_adj(edge_index, batch)        
        out = self.transformer_layer(batch_x, mask=mask, att_bias=adj)
        out = out[mask]
        return out
    
    # TODO: data => [x, edge_index, batch]
    def forward(self,x, edge_index, batch):
        '''
            input:  hidden_dim
            output: hidden_dim
        '''
        # x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        x_fusion = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        
        x_gcn = self.GCN_layer(x, edge_index)
        x_trans = self.graphTrans_fc(x, edge_index, batch)
        x_gcn_mix = x_gcn * self.alpha
        x_trans_mix = x_trans * ( 1.00 - self.alpha )
        x_fusion = torch.add(x_gcn_mix, x_trans_mix)

        return x_fusion

'''
    return multi-header in this layer:  [gcn, transformer, fusion]
'''
class SelfMultiFusionLayer(nn.Module):
    def __init__(self, args, position="s"):
        super(SelfMultiFusionLayer, self).__init__()
        #* gcn_channels is for multi-hop [1, gcn_channels] gcn kernels 
        self.args = args

        self.features       = args.in_size
        self.hidden_dim     = args.hidden_dim
        self.num_layers     = args.gcn_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.dropout
        self.alpha          = args.alpha
        self.device         = args.device
        self.pos_enc        = args.pos_encoding
        self.pos_embedding  = PositionEncoder(self.args, self.hidden_dim, pos_enc=self.pos_enc,
                                                embedding_type=position)
        
        #* GCN*1 + Trans*1
        # hidden -> hidden
        self.GCN_layer = MultiScaleGCNLayer(
            gcn_channels=args.gcn_channels,
            in_size=args.gcn_hidden,       # mid stage so the input is hidden_dim
            hidden_dim=args.gcn_hidden,
            out_dim=args.gcn_hidden,
            dropout=args.dropout,
            device=self.device
        )
        self.transformer_layer = TransformerEncoder(self.args)
        self.fc1 = nn.Linear(self.features, self.features)

    
    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        self.GCN_layer.reset_parameters()
        self.transformer_layer.reset_parameters()
        self.fc1.reset_parameters()
    
    def reset_alpha(self, alpha):
        assert alpha <= 1
        self.alpha = alpha

    # input => x_trans 
    def graphTrans_fc(self, x, edge_index, batch):
        out = self.pos_embedding(x, edge_index)
        batch_x, mask = to_dense_batch(out, batch)
        adj = to_dense_adj(edge_index, batch)        
        out = self.transformer_layer(batch_x, mask=mask, att_bias=adj)
        out = out[mask]
        return out
    

    def forward(self, x_gcn, x_trans, x_fusion, 
                edge_index, batch):
        '''
            NOTE that gcn and trans can be viewd as independent module.
            input:  hidden_dim  [x_gcn, x_trans, x_mix]
            output: hidden_dim  [x_gcn, x_trans, x_mix]
        '''

        x_fusion = torch.zeros(x_fusion.size(0), self.hidden_dim).to(self.device)
        
        x_gcn = self.GCN_layer(x_gcn, edge_index)
        x_trans = self.graphTrans_fc(x_trans, edge_index, batch)
        x_gcn_mix = x_gcn * self.alpha
        x_trans_mix = x_trans * ( 1.00 - self.alpha )
        x_fusion = torch.add(x_gcn_mix, x_trans_mix)

        return x_gcn, x_trans, x_fusion


'''
Below are modulers consists of simple layers. (can be seen as uni-modal encoders)
GCN, transformer, fusion model may be included.
'''

'''
    K-hop GCNs for early && late fusion
'''
class MultiScaleGCN(nn.Module):
    def __init__(self, args):
        super(MultiScaleGCN, self).__init__()
        # type -> early for early fusion && 
        #      -> late for late fusion
        self.fusion_type    = args.fusion_type
        self.gcn_channels   = args.gcn_channels
        self.num_features   = args.in_size
        self.hidden_dim     = args.gcn_hidden
        self.num_layers     = args.gcn_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.gcn_dropout
        self.device         = args.device
                
        assert(self.gcn_channels >= 1)
        
        #* composition of SGC layers (Layers*4)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers-1):
            self.convs.append( MultiScaleGCNLayer(
                self.gcn_channels,
                self.num_features,
                self.hidden_dim,
                self.num_features,
                self.dropout,
                self.device
            ) )
    
        self.fc1 = torch.nn.Linear(self.num_features, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)
    
    def fc_forward(self, x):
        '''
            out_dim -> final output
        '''
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x
    
    
    def forward(self, data):
        '''
        @description:   given a graph representation, we hope to split it into 
                    x_1, x_2, ... x_n, and concat them finally.
        @param:         data for pyg.type data 
        @return:        mean value of [x_1, x_2, ... x_n]   
        '''
        data_temp = data
        x, edge_index, batch = data.x, data.edge_index, data.batch

        X_k = copy.deepcopy(x)     
        for conv in self.convs:
            # data_temp.x = X_k
            X_k = conv(X_k, edge_index)
            # X_k = conv(data_temp)

        X_k = torch.div(X_k, self.gcn_channels)     
        X_k = global_add_pool(X_k, batch)
        X_k = self.fc_forward(X_k)
        return X_k

    def __repr__(self):
        return self.__class__.__name__


class GraphTransformer(nn.Module):
    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.pos_encoding = args.pos_encoding
        self.pos_embedding = PositionEncoder(self.args,
                                            self.hidden_dim,
                                            pos_enc=self.pos_encoding)
        self.encoder1 = TransformerEncoder(self.args)
        self.encoders = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.encoders.append(TransformerEncoder(args))
        
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.num_classes)
    
    
    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        print("Start")
        print(x.size(), edge_index.size())
        # BETTER NOT USE THIS MODULE
        x = self.pos_embedding(x, edge_index)
        print(x.size())
        batch_x, mask = to_dense_batch(x, batch)
        print(batch_x.size())
        adj = to_dense_adj(edge_index, batch)   
        print(edge_index.size())
        x = self.encoder1(batch_x, mask=mask, att_bias=adj)
        print(x.size())
        for encoder in self.encoders:
            x = encoder(x, mask, att_bias=adj)
        x = x.mean(dim=1)
        out = F.relu( self.fc1(x) )
        out = F.dropout(out, p=self.dropout, training=self.training)
        print("End")

        return out


'''
    Graph mixup-fusion encoder
'''
class GraphMixupFusion(nn.Module):
    def __init__(self, args) -> None:
        super(GraphMixupFusion, self).__init__()
        self.args = args
        self.num_layers = args.num_fusion_layers
        # gcn layer parameters
        self.fusion_type    = args.fusion_type
        self.gcn_channels   = args.gcn_channels
        self.num_features   = args.in_size
        self.hidden_dim     = args.gcn_hidden
        self.num_layers     = args.gcn_layers
        self.num_classes    = args.num_classes
        self.dropout        = args.gcn_dropout
        self.device         = args.device
        # trans layer parameters
        self.pos_encoding   = args.pos_encoding
        self.pos_embedding = PositionEncoder(self.args,
                                            self.hidden_dim,
                                            pos_enc=self.pos_encoding)
        self.hidden_pos_embedding = PositionEncoder(self.args,
                                                    self.hidden_dim,
                                                    pos_enc=self.pos_encoding,
                                                    embedding_type="m")
        self.alpha          = args.alpha
        
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
        for i in range(self.num_layers-1):
            alpha_1 = np.random.normal(loc=self.alpha, scale=self.alpha)
            if alpha_1 >= 1:
                alpha_1 = self.alpha
            self.encoders.append(SelfMixupFusionLayer(self.args, position="m"))
            self.encoders[i].reset_alpha(alpha_1)

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
        #* TODO, here, tensor.shape 2 -> 3
        x_gcn1 = self.gcn_encoder1(x, edge_index)
        x_trans1 = self.trans_encoder1(batch_x, mask=mask, att_bias=adj)
        x_trans1 = x_trans1[mask]

        x_gcn1 = x_gcn1 * self.alpha
        x_trans1 = x_trans1 * (1.00 - self.alpha)
        
        x = torch.add(x_gcn1, x_trans1)
        for encoder in self.encoders:
            x = encoder(x, edge_index, batch) # TODO
        
        x = global_add_pool(x, batch)
        out = self.fc_forward(x)
        return out



#! BUG HERE and I don't want to modify cause graphormer is a piece of shit.
class GraphormerEncoder(nn.Module):
    def __init__(self, args):
        super(GraphormerEncoder, self).__init__()
        self.num_layers = args.trans_num_layers
        self.input_node_dim = args.input_node_dim
        self.hidden_node_dim = args.hidden_node_dim
        self.input_edge_dim = args.input_edge_dim
        self.hidden_edge_dim = args.hidden_edge_dim
        self.output_node_dim = args.output_dim
        self.n_heads = args.n_heads
        
        self.max_in_degree = args.max_in_degree
        self.max_out_degree = args.max_out_degree
        self.max_path_distance = args.max_path_distance
        
        self.fc_node_in = torch.nn.Linear(self.input_node_dim, self.hidden_node_dim)
        self.fc_edge_in = torch.nn.Linear(self.input_edge_dim, self.hidden_edge_dim)
        self.spatial_encoding = SpatialEncoding(
            max_path_distance = self.max_path_distance,
        )
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.hidden_node_dim,
                edge_dim=self.hidden_edge_dim,
                n_heads=self.n_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)            
        ])
        self.fc_node_out = torch.nn.Linear(self.hidden_node_dim, self.output_node_dim)
        
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()
        
        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)
        
        x = self.fc_node_in(x)
        edge_attr = self.fc_edge_in(edge_attr)
        b = self.spatial_encoding(x, node_paths)
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)
        x = self.fc_node_out(x)
        
        x = global_mean_pool(x, data.batch)
        
        return x