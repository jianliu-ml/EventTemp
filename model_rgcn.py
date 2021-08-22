import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from dgl import DGLGraph

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import RelGraphConv
from functools import partial

import torch.nn as nn

class BaseRGCN(nn.Module):
    def __init__(self, ipt_dim, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False):
        super(BaseRGCN, self).__init__()
        self.ipt_dim = ipt_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_size, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, hidden_size)
#         self.conv2 = GraphConv(hidden_size, num_classes)

#     def forward(self, g, inputs):
#         h = self.conv1(g, inputs)
#         h = torch.relu(h)
#         h = self.conv2(g, h)
#         return h

class RGCN(BaseRGCN):

    def build_input_layer(self):
        return RelGraphConv(self.ipt_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)
    
    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=None,
                self_loop=self.use_self_loop)