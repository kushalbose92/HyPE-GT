import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features for ZINC dataset
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

from hyp_pos_enc import PositionalEncoding

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.c = net_params['c']
        self.act = net_params['act']
        self.pe_layers = net_params['pe_layers']
        self.manifold = net_params['manifold']
        self.model = net_params['model']
        max_wl_role_index = 37 # this is maximum graph size in the dataset

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.pos_enc_model = PositionalEncoding(self.c, self.act, self.pe_layers, self.manifold, self.model, hidden_dim, 0.0, self.device)
        self.pos_enc_model.to(self.device)    
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(self.n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, g, h, e, pos_enc):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
       
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)

        p = self.hyp_positional_encodings(g, pos_enc, self.pos_enc_model)
        e = self.embedding_e(e)   
        h = h + p 
        
        # convnets
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
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        adj = g.adj().to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold == 'Hyperboloid':
            hyp_pos_feat = hyp_pos_feat[:, 1:]

        g.ndata['pos_enc'] = hyp_pos_feat

        return g.ndata['pos_enc']
