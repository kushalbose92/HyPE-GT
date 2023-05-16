import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    Graph Transformer with edge features for OGBG-MOLHIV
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

from hyp_pos_enc import PositionalEncoding

class GraphTransformer(nn.Module):
    def __init__(self, net_params):
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

        self.pos_enc_model = PositionalEncoding(self.c, self.act, self.pe_layers, self.manifold, self.model, GT_hidden_dim, dropout, self.device)
        self.pos_enc_model.to(self.device)  
        self.embedding_p = nn.Linear(self.pe_dim, GT_hidden_dim)
        
        self.embedding_h = AtomEncoder(emb_dim = GT_hidden_dim) 
        self.embedding_e = BondEncoder(emb_dim = GT_hidden_dim)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(GT_out_dim, 1)   # 1 out dim for probability      
        

    def forward(self, g, h, e, pos_enc):
        

        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        h = self.in_feat_dropout(h)

        # add hyperbolic PE here
        p = self.hyp_positional_encodings(g, pos_enc, self.pos_enc_model) 
        h = h + p
    
        # Second Transformer
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
   
        sig = nn.Sigmoid()
    
        return sig(self.MLP_layer(hg))
        
    def loss(self, scores, targets):
        
        loss = nn.BCELoss()
        
        l = loss(scores.float(), targets.float())
        
        return l

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        adj = g.adj().to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold=='Hyperboloid':
            hyp_pos_feat = hyp_pos_feat[:, 1:]

        g.ndata['pos_enc'] = hyp_pos_feat

        return g.ndata['pos_enc']
