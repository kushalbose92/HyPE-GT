import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers
import manifolds
from torch.nn.modules.module import Module


class HNN(nn.Module):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, act, num_layers, manifold, dim, dropout, device):
        super(HNN, self).__init__()
        self.c = c
        self.manifold = getattr(manifolds, manifold)()
        # assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(c, act, num_layers, manifold, dim, device)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, dropout, act, 1, device)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        h = self.layers(x_hyp)
        return h

class HGCN(nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, act, num_layers, manifold, dim, dropout, device):
        super(HGCN, self).__init__()
        self.c = c
        self.manifold = getattr(manifolds, manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(c, act, num_layers, manifold, dim, device)
        self.dims = dims
        self.acts = acts
        self.device = device
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, dropout, act, 1, device))
        self.layers = nn.Sequential(*hgc_layers)

    def encode(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        h, adj = self.layers((x_hyp, adj))
        return h


# class LinearDecoder(nn.Module):
#     """
#     MLP Decoder for Hyperbolic/Euclidean node classification models.
#     """

#     def __init__(self, c, args):
#         super(LinearDecoder, self).__init__()
#         self.manifold = getattr(manifolds, args.manifold)()
#         self.input_dim = args.dim
#         self.output_dim = 7
#         self.bias = 1
#         self.dropout = args.dropout
#         self.c = c
#         self.cls = nn.Linear(self.input_dim, self.output_dim)

#     def decode(self, x, adj):
#         h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
#         hidden = self.cls(h)
#         hidden = F.dropout(hidden, self.dropout, training=self.training)
#         return hidden



class PositionalEncoding(nn.Module):
    def __init__(self, c, act, num_layers, manifold, model, dim, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.c = c
        self.manifold = manifold 
        self.encoder = None

        if model=='HGCN':
            # print("Using: Hyperbolic GCN")
            self.encoder = HGCN(c, act, num_layers, manifold, dim, dropout, device)
        elif model=='HNN':
            # print("Using: Hyperbolic Neural Network")
            self.encoder = HNN(c, act, num_layers, manifold, dim, dropout, device)
        else:
            print("Invaid model name")
        # self.decoder = LinearDecoder(self.c, args)
        
        if self.c is not None:
            self.c = torch.tensor([c])
            if not device == -1:
                self.c = self.c.to(device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold)()

        if self.manifold.name == 'Hyperboloid':
            dim = dim + 1
        
    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h
        
    # def decode(self, h, adj, idx):
    #     output = self.decoder.decode(h, adj)
    #     return F.log_softmax(output[idx], dim=1)
    

