import dgl
import torch
import torch.nn.functional as F
import numpy as np 
from scipy import sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# laplacian positional encodings
def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/s Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g.ndata['pos_enc']

# random walk positional encodings
def rand_walk_positional_encoding(g, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    # Geometric diffusion features with Random Walk
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
    RW = A * Dinv  
    M = RW
    
    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE,dim=-1)
    g.ndata['pos_enc'] = PE  
    
    return g.ndata['pos_enc']

def loss_fn(pred, label):
    
    return F.cross_entropy(pred, label)

def adj_normalization(data):
    
  adj_matrix = torch.zeros(data.num_nodes, data.num_nodes)

  for e in range(data.num_edges):
    src = data.edge_index[0][e]
    tgt = data.edge_index[1][e]
    adj_matrix[src][tgt] = 1

  # normalization 
  adj_matrix += torch.eye(data.num_nodes)
  degrees = torch.sum(adj_matrix, dim = 1)
  degree_matrix = torch.diag(1 / torch.sqrt(degrees))
  norm_adj = torch.mm(degree_matrix, adj_matrix)
  norm_adj = torch.mm(norm_adj, degree_matrix)

  return norm_adj

#  visualize node embeddings
def visualize(feat_map, color, data_name, num_layers):
    z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(os.getcwd() + "/visuals/" + data_name + "_" + str(num_layers) + "_embedding.png")
    plt.clf()
