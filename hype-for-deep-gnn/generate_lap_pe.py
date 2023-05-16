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


    
