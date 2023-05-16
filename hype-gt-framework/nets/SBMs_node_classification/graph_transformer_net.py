import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer for PATTERN, CLUSTER
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

from hyp_pos_enc import PositionalEncoding

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.c = net_params['c']
        self.act = net_params['act']
        self.pe_layers = net_params['pe_layers']
        self.manifold = net_params['manifold']
        self.model = net_params['model']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.pos_enc_model = PositionalEncoding(self.c, self.act, self.pe_layers, self.manifold, self.model, hidden_dim, dropout, self.device)
        self.pos_enc_model.to(self.device)    
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)


    def forward(self, g, h, e, pos_enc, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        e = self.embedding_e(e) 
        
        p = self.hyp_positional_encodings(g, pos_enc, self.pos_enc_model)  
        h = h + p

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
            
        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    def hyp_positional_encodings(self, g, pe_init, pos_enc_model):

        pos_feat = self.embedding_p(pe_init)
        adj = g.adj().to(self.device)
        hyp_pos_feat = pos_enc_model.encode(pos_feat, adj)

        if self.pe_layers == 1 and self.manifold == 'Hyperboloid':
            hyp_pos_feat = hyp_pos_feat[:, 1:]

        g.ndata['pos_enc'] = hyp_pos_feat

        return g.ndata['pos_enc']




        
