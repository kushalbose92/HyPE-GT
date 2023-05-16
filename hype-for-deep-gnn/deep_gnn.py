import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

from hyp_pos_enc import PositionalEncoding
import manifolds
import layers.hyp_layers as hyp_layers
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge, GCN2Conv
from torch_geometric.utils import to_dense_adj 


# Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, run_base, device):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.device = device
        self.run_base = run_base
        
        self.gcn_convs = nn.ModuleList()
        if self.run_base == False:
            self.manifold = getattr(manifolds, self.manifold)()
            self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
            self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim, catched=True))

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)
        

    def forward(self, x_h, adj, edge_index, pos_feat):
    
        # (N, d)
        if self.run_base == False:
            pos_feat = self.pos_embedding(pos_feat)
            hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

            if self.num_layers == 1:
                hyp_pos_feat = hyp_pos_feat[:, 1:]
            hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, edge_index)

            if i != self.gcn_layers-1:
                
                if self.run_base == False:
                    
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    x_h = x_h + hyp_pos_feat
                
                # fractional concatenation
                # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)
    
        # x_h = x_h + hyp_pos_feat
        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h

# Graph Attention Networks
class GAT(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, device):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.device = device
        self.alpha = 0.1
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        self.manifold = getattr(manifolds, self.manifold)()
        self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(GATConv(self.hidden_dim, self.hidden_dim))

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)
        
        self.pos_embedding.reset_parameters()
        self.init_w.reset_parameters()
        self.last_w.reset_parameters()
        

    def forward(self, x_h, adj, edge_index, pos_feat, run_base):
    
        # (N, d)
        pos_feat = self.pos_embedding(pos_feat)
        hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

        if self.num_layers == 1:
            hyp_pos_feat = hyp_pos_feat[:, 1:]
        hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, edge_index)

            if i != self.gcn_layers-1:

                if run_base == 'False':
                
                    '''
                    strategy 1
                    '''    
                    x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    # x_h = x_h + hyp_pos_feat
                
                # fractional concatenation
                # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)
    
        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h

# Graph Sample and Aggregate
class SAGE(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, device):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.device = device
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        self.manifold = getattr(manifolds, self.manifold)()
        self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(SAGEConv(self.hidden_dim, self.hidden_dim, aggr= 'mean'))
            self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

    def forward(self, x_h, adj, edge_index, pos_feat, run_base):
    
        # (N, d)
        pos_feat = self.pos_embedding(pos_feat)
        hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

        if self.num_layers == 1:
            hyp_pos_feat = hyp_pos_feat[:, 1:]
        hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, edge_index)

            if i != self.gcn_layers-1:

                if run_base == 'False':
                    
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    x_h = x_h + hyp_pos_feat
                
                # fractional concatenation
                # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)
    
        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h

# Graph Isomorphism Network
class GIN(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, device):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.device = device
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        self.manifold = getattr(manifolds, self.manifold)()
        self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.mid_w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

        for i in range(self.gcn_layers):
            if i == 0:
                self.gcn_convs.append(GINConv(self.init_w, eps=0.1, train_eps=True))
            elif i == self.gcn_layers-1:
                self.gcn_convs.append(GINConv(self.last_w, eps=0.1, train_eps=True))
            else:
                self.gcn_convs.append(GINConv(self.mid_w, eps=0.1, train_eps=True))
            self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

        
    def forward(self, x_h, adj, edge_index, pos_feat, run_base):
    
        # (N, d)
        pos_feat = self.pos_embedding(pos_feat)
        hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

        if self.num_layers == 1:
            hyp_pos_feat = hyp_pos_feat[:, 1:]
        hyp_pos_feat = F.tanh(hyp_pos_feat)

        # x_h = self.init_w(x_h)

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, edge_index)

            if i != self.gcn_layers-1:

                if run_base == 'False':
                    
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    x_h = x_h + hyp_pos_feat
                
                # fractional concatenation
                # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)
    
        # x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h


# Jumping Knowledge Networks
class JKNet(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, run_base, device):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.device = device
        self.run_base = run_base
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        if run_base == False:
            self.manifold = getattr(manifolds, self.manifold)()
            self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
            self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim, cached=True))
            self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

        
    def forward(self, x_h, adj, edge_index, pos_feat):
    
        # (N, d)
        if self.run_base == False:
            pos_feat = self.pos_embedding(pos_feat)
            hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

            if self.num_layers == 1:
                hyp_pos_feat = hyp_pos_feat[:, 1:]
            hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)

        layer_outputs = []
        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, edge_index)

            if i != self.gcn_layers-1:

                if self.run_base == False:
                    
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    # x_h = x_h + hyp_pos_feat
                
                # fractional concatenation
                # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)

                layer_outputs.append(x_h)
    
        x_h = torch.stack(layer_outputs, dim=0)
        x_h = torch.max(x_h, dim=0)[0]

        if self.run_base == False:
            x_h = x_h + hyp_pos_feat

        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h


# Simple and Deep GCN - GCNII
class GCNII(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, run_base, device):
        super(GCNII, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.alpha = 0.1
        self.theta = 0.2
        self.beta = 0.3
        self.device = device
        self.run_base = run_base
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        if run_base == False:
            self.manifold = getattr(manifolds, self.manifold)()
            self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
            self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(GCN2Conv(self.hidden_dim, self.theta, self.beta, i+1))
            self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

    def forward(self, x_h, adj, edge_index, pos_feat):
    
        # (N, d)
        if self.run_base == False:
            pos_feat = self.pos_embedding(pos_feat)
            hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

            if self.num_layers == 1:
                hyp_pos_feat = hyp_pos_feat[:, 1:]
            hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)
        x_0 = x_h

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, x_0, edge_index)

            if i != self.gcn_layers-1:
                
                if self.run_base == False:
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    # x_h = x_h + hyp_pos_feat
                    
                    # fractional concatenation
                    # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                    # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                    # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)

        x_h = x_h + hyp_pos_feat
        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h


class PNA(nn.Module):
    def __init__(self, data_obj, gcn_layers, c, act, num_layers, manifold, model, pos_enc_dim, hidden_dim, dropout, device):
        super(GCNII, self).__init__()
        self.num_layers = num_layers
        self.c = c
        self.data_obj = data_obj
        self.pos_enc_dim = pos_enc_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn_layers = gcn_layers
        self.manifold = manifold
        self.alpha = 0.1
        self.theta = 0.2
        self.beta = 0.3
        self.device = device
        
        self.gcn_convs = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        self.manifold = getattr(manifolds, self.manifold)()
        self.pos_embedding = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.pos_enc_model = PositionalEncoding(c, act, num_layers, manifold, model, hidden_dim, dropout, device)

        for i in range(self.gcn_layers):
            self.gcn_convs.append(GCN2Conv(self.hidden_dim, self.theta, self.beta, i+1))
            self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

        self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
        self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

    def forward(self, x_h, adj, edge_index, pos_feat, run_base):
    
        # (N, d)
        pos_feat = self.pos_embedding(pos_feat)
        hyp_pos_feat = self.pos_enc_model.encode(pos_feat, adj)

        if self.num_layers == 1:
            hyp_pos_feat = hyp_pos_feat[:, 1:]
        hyp_pos_feat = F.tanh(hyp_pos_feat)

        x_h = self.init_w(x_h)
        x_0 = x_h

        for i in range(self.gcn_layers):
            x_h = self.gcn_convs[i](x_h, x_0, edge_index)

            if i != self.gcn_layers-1:
                
                if run_base == 'False':
                    '''
                    strategy 1
                    '''    
                    # x_h = (x_h * (1 - self.alpha)) + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 2
                    '''
                    # x_h = x_h + (hyp_pos_feat * self.alpha)
                    
                    '''
                    strategy 3
                    '''
                    x_h = x_h + hyp_pos_feat
                    
                    # fractional concatenation
                    # frac_hyp_feat = (self.alpha * hyp_pos_feat)
                    # x_h = torch.cat([x_h, frac_hyp_feat], dim = 1)
                    # x_h = self.lin_layers[i](x_h)
                
                x_h = F.dropout(x_h, p=self.dropout, training=self.training)
                x_h = F.relu(x_h)
    
        x_h = self.last_w(x_h)
                
        embedding = x_h
        x_h = F.log_softmax(x_h, dim=1)

        return embedding, x_h



