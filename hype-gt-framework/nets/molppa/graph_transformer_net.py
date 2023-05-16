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
        self.MLP_layer = MLPReadout(out_dim, 37)   # 1 out dim since regression problem       
        
    def forward(self, graph, h, e, pos_enc):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)

        p = self.hyp_positional_encodings(graph, pos_enc, self.pos_enc_model)   
        h = h + p 
        
        # convnets
        g = dgl.graph((graph.edge_index[0], graph.edge_index[1])).to(self.device)
        for conv in self.layers:
            h, e = conv(g, h, e)
        # g.ndata['h'] = h

        # if self.readout == "sum":
        #     hg = dgl.sum_nodes(g, 'h')
        # elif self.readout == "max":
        #     hg = dgl.max_nodes(g, 'h')
        # elif self.readout == "mean":
        #     hg = dgl.mean_nodes(g, 'h')
        # else:
            # hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        h = gnn.global_mean_pool(h, graph.batch) 
            
        h_out = self.MLP_layer(h)
        return h_out
        
    def loss(self, scores, targets):
        loss = nn.CrossEntropyLoss()(scores, targets.squeeze())
        return loss

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        adj = to_dense_adj(g.edge_index).squeeze(0)
        adj = adj.to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold == 'Hyperboloid':
            hyp_pos_feat = hyp_pos_feat[:, 1:]

        return hyp_pos_feat
