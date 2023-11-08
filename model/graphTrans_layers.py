import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import Linear, BatchNorm1d as BN, Sequential, ReLU
from torch_geometric.nn import GINConv, GCNConv


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


class FeedForwardNetwork(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hid)
        self.lin2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.lin2(self.gelu(self.lin1(x)))
        x = self.dropout(x)
        x += residual

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.self_attention = MultiHeadAttention(args)
        self.ffn = FeedForwardNetwork(self.d_model, self.d_model)

    def reset_parameters(self):
        self.self_attention.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x, mask=None, att_bias=None):
        output_att = self.self_attention(x, mask=mask, att_bias=att_bias)
        output_ffn = self.ffn(output_att)
        return output_ffn

""" Override transformer block
"""
class FusionTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim


class PositionEncoder(nn.Module):
    '''
        NUM LAYER == 3 MAYBE NOT A GOOD CHOICE.
    '''
    def __init__(self, args, hidden, pos_enc=None, num_layers=3, embedding_type="s"):
        super(PositionEncoder, self).__init__()
        self.dropout = args.dropout
        self.num_layers = num_layers
        self.pos_enc = pos_enc
        self.hidden = hidden
        self.embedding_type = embedding_type
        # define embedding module
        # if 'ogb' in args.dataset:
        #     self.embedding = AtomEncoder(hidden)
        # else:
        #     self.embedding = Linear(args.num_features, hidden)
        if self.embedding_type == "m":
            self.embedding = torch.nn.Linear(args.hidden_dim, hidden)
        else:         
            self.embedding = torch.nn.Linear(args.num_features, hidden)
        # define positional encoding module
        if pos_enc == 'gcn':
            self.conv1 = GCNConv(self.hidden, self.hidden)
            self.convs = torch.nn.ModuleList()
            for i in range(self.num_layers - 1):
                self.convs.append(GCNConv(self.hidden, self.hidden))
        elif pos_enc == 'gin':
            self.conv1 = GINConv(
                Sequential(
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BN(self.hidden),
                ), train_eps=False)
            self.convs = torch.nn.ModuleList()
            for i in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(self.hidden, self.hidden),
                            ReLU(),
                            Linear(self.hidden, self.hidden),
                            ReLU(),
                            BN(self.hidden),
                        ), train_eps=False))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        if self.pos_enc is not None:
            x = self.embedding(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x, inplace=True)
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = F.relu(x, inplace=True)

        return x

    def __repr__(self):
        return self.__class__.__name__
