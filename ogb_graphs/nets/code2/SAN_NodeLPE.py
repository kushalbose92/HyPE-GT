import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    Graph Transformer with edge features for ogbg-code2 dataset

"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

from hyp_pos_enc import PositionalEncoding

class GraphTransformer(nn.Module):
    def __init__(self, net_params, node_encoder):
        super().__init__()

        full_graph = net_params['full_graph']
        gamma = net_params['gamma']

        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']

        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.c = net_params['c']
        self.act = net_params['act']
        self.pe_layers = net_params['pe_layers']
        self.manifold = net_params['manifold']
        self.model = net_params['model']
        self.pe_dim = net_params['pe_dim']
        self.emb_dim = net_params['emb_dim']

        self.num_vocab = net_params['num_vocab']
        self.max_seq_len = net_params['max_seq_len']
        self.node_encoder = node_encoder
        self.embedding_e = nn.Linear(2, self.emb_dim)

        self.pos_enc_model = PositionalEncoding(self.c, self.act, self.pe_layers, self.manifold, self.model, GT_hidden_dim, dropout, self.device)
        self.pos_enc_model.to(self.device)  
        self.embedding_p = nn.Linear(self.pe_dim, GT_hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])

        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(GT_out_dim, 128)   # 1 out dim for probability

        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.readout == "set2set":
            for i in range(self.max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(2*self.emb_dim, self.num_vocab))
        else:
            for i in range(self.max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_vocab))


    def forward(self, g, h, e, pos_enc):

        # input embedding
        h = [self.node_encoder(h, g.node_depth.view(-1,))]
        h = h[0].to(self.device)
        h = self.in_feat_dropout(h)
        
        g_dgl = dgl.graph((g.edge_index[0], g.edge_index[1]))
        e = self.embedding_e(e)
        
        # add hyperbolic PE here
        h = self.add_hyp_pe(g_dgl, h, self.pos_enc_model, pos_enc, self.c)
        # pos_enc = self.embedding_p(pos_enc)
        # h = h + pos_enc

        # Second Transformer
        for conv in self.layers:
            h, e = conv(g_dgl, h, e)
        # print(h.shape)

        if self.readout == "sum":
            hg = global_add_pool(h, g.batch)
        elif self.readout == "max":
            hg = global_max_pool(h, g.batch)
        elif self.readout == "mean":
            hg = global_mean_pool(h, g.batch)
        else:
            hg = global_mean_pool(h, g.batch)  # default readout is mean nodes

        # print("batch ", hg.shape)
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](hg))

        return pred_list


    # def loss(self, scores, targets):

    #     loss = nn.BCELoss()

    #     l = loss(scores, targets.to(dtype=scores.dtype))

    #     return l

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        # adj = g.adj().to(self.device)
        adj = g.adj_external().to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold=='Hyperboloid':
            hyp_pos_feat = hyp_pos_feat[:, 1:]

        g.ndata['pos_enc'] = hyp_pos_feat

        return g.ndata['pos_enc']


    # HyPE-GT
    def add_hyp_pe(self, g, h, pos_enc_model, pos_enc, c):
    
        # generation of hyperbolic positional encodings
        p = self.hyp_positional_encodings(g, pos_enc, pos_enc_model)  

        # map h to hyperbolic space
        h_hyp = pos_enc_model.map_euc_feat_to_hyp(h)

        # add with PE
        h_hyp = pos_enc_model.mobius_add_with_pe(h_hyp, p, c)

        # revert back h to euclidean space
        h = pos_enc_model.map_hyp_feat_to_euc(h_hyp)

        return h

