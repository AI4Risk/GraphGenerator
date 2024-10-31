import sys
import numpy as np
from re import T
from numpy import eye
import torch
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from .layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
DTYPE = torch.float32


class EmbeddingEncoder(nn.Module):
    def __init__(self, N, H, pretrained_emb=None):
        super(EmbeddingEncoder, self).__init__()
        self.encoding = nn.Embedding(N, H)
        if pretrained_emb is not None:
            print('Using pretrained node embedding!')
            self.encoding.weight.data.copy_(torch.tensor(pretrained_emb))
            self.encoding.weight.requires_grad = True
    def forward(self, pretrained_emb=None):
        if pretrained_emb is not None:
            print('Using pretrained node embedding!')
            self.encoding.weight.data.copy_(pretrained_emb)
            self.encoding.weight.requires_grad = True
        return self.encoding.weight


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
        

class LinearDecoder(nn.Module):
    def __init__(self, N, H):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(H, N)
    def forward(self, X):
        # X: N_all * H
        #    T * N * H
        # Linear: H * N
        # output: N_all * N
        return self.decoder(X)


class NonLinearDecoder(nn.Module):
    def __init__(self, N, H):
        super(NonLinearDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(H, H),
                                     nn.ReLU(),
                                     nn.Linear(H, N))
    def forward(self, X):
        return self.decoder(X)


class GATLayer(nn.Module):
    def __init__(self, in_dim=128, hid_dim=32, n_heads=4):
        super(GATLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.emb_src = nn.Linear(in_dim, hid_dim * n_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_heads, hid_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_heads, hid_dim)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.skip_feat = nn.Linear(in_dim, hid_dim * n_heads)
        self.gate = nn.Linear(3 * hid_dim * n_heads, 1)
        self.norm = nn.LayerNorm(hid_dim * n_heads)
        self.activation = nn.PReLU(init=0.25)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):
        feat_src = self.emb_src(feat).view(-1, self.n_heads, self.hid_dim)
        feat_dst = feat_src[:graph.number_of_dst_nodes()]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft'].reshape(-1, self.hid_dim*self.n_heads)
        skip_feat = self.skip_feat(feat)[:graph.number_of_dst_nodes()]
        gate = torch.sigmoid(self.gate(torch.cat([rst, skip_feat, rst - skip_feat], dim=-1)))
        rst = gate * rst + (1 - gate) * skip_feat
        return self.activation(self.norm(rst))


class ScalableGAE(nn.Module):
    def __init__(self, in_dim=128, hid_dim=32, n_heads=4, out_dim=128):
        super(ScalableGAE, self).__init__()
        # self.input_encoder = nn.Linear(out_dim, in_dim)
        self.attention_encoder = GATLayer(in_dim=in_dim, hid_dim=hid_dim, n_heads=n_heads)
        self.decoder = nn.Linear(n_heads * hid_dim, out_dim)

    def forward(self, blocks, feat):
        return self.decoder(self.attention_encoder(blocks[0], feat))


def coo_to_csp(sp_coo):
    row = sp_coo.row
    col = sp_coo.col
    
    sp_tensor = torch.sparse_coo_tensor(torch.tensor(np.stack([row, col])), 
                                        torch.tensor(sp_coo.data), sp_coo.shape)
    
    return sp_tensor
