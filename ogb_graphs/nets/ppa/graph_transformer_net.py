import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_adj

import dgl

"""
    Graph Transformer with edge features for OGBG-PPA dataset
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from hyp_pos_enc import PositionalEncoding

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        # in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        # n_classes = net_params['n_classes']
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

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.pos_enc_model = PositionalEncoding(self.c, self.act, self.pe_layers, self.manifold, self.model, hidden_dim, dropout, self.device)
        self.pos_enc_model.to(self.device)    
        
        self.embedding_h = nn.Linear(7, hidden_dim)
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Linear(7, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(self.n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, 37)    

        
    def forward(self, g, h, e, pos_enc):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)

        dgl_g = dgl.graph((g.edge_index[0], g.edge_index[1])).to(self.device)

        # adding learnable hyperbolic positional encodings
        # h = self.add_hyp_pe(dgl_g, h, self.pos_enc_model, pos_enc, self.c)
        h = h + pos_enc
        
        # convnets
        for conv in self.layers:
            h, e = conv(dgl_g, h, e)

        if self.readout == "sum":
            hg = gnn.global_add_pool(h, g.batch)
        elif self.readout == "max":
            hg = gnn.global_max_pool(h, g.batch)
        elif self.readout == "mean":
            hg = gnn.global_mean_pool(h, g.batch)
        else:
            hg = gnn.global_mean_pool(h, g.batch)  # default readout is mean nodes
     
        h_out = self.MLP_layer(hg)
        # print("out ", h_out.shape)
        return h_out
        
    def loss(self, scores, targets):
        loss = nn.CrossEntropyLoss()(scores, targets.squeeze())
        return loss

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        # adj = g.adj().to(self.device)
        adj = g.adj_external().to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold == 'Hyperboloid':
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

