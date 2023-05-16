import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random
import dgl 
from scipy import sparse as sp
import numpy as np 
import os

class CoauthorPhysics():

    def __init__(self):

        dataset = Coauthor(root='data/coauthor_physics', name='Physics', transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = "CoauthorPhysics"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        # self.train_label_rate = (int(self.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.has_isolated_nodes()
        self.data_contains_self_loops = data.has_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index
        # self.train_mask = data.train_mask

        train_idx, val_idx, test_idx = self.index_generation()
        self.train_mask = self.mask_generation(train_idx, self.num_nodes)
        self.val_mask = self.mask_generation(val_idx, self.num_nodes)
        self.test_mask = self.mask_generation(test_idx, self.num_nodes)

    # adjacency list generation
    def adj_list_generation(self, edge_index):

        adj_list = [[] for n in range(self.num_nodes)]
        src_list = edge_index[0]
        dest_list = edge_index[1]

        for n in range(self.num_edges):

            adj_list[int(src_list[n])].append(int(dest_list[n]))

        return adj_list

    def index_generation(self):

        class_idx = [[] for i in range(self.num_classes)]
        train_idx = []
        val_idx = []
        test_idx = []
        for n in range(self.num_nodes):
            
            class_idx[self.node_labels[n]].append(n)

        # z = [len(class_idx[i]) for i in range(len(class_idx))]
        # print(z)
        all_indices = []
        for c in range(self.num_classes):

            sampled_c = random.sample(class_idx[c], len(class_idx[c])) 
            random.shuffle(sampled_c)
            train_set = sampled_c[:20]
            all_indices += sampled_c[20:]
            train_idx += train_set
    
        random.shuffle(all_indices)
        val_idx = all_indices[:150]
        test_idx = all_indices[150:]

        return train_idx, val_idx, test_idx

    def mask_generation(self, index, num_nodes):
        mask = torch.zeros(num_nodes, dtype = torch.bool)
        mask[index] = 1
        return mask
    def lap_positional_encoding(self, g):
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
        g.ndata['pos_enc'] = torch.from_numpy(EigVec).float() 
        
        return g.ndata['pos_enc']


# obj = CoauthorPhysics()
# edge_index = obj.edge_index
# g = dgl.graph((edge_index[0], edge_index[1]))
# lap_pe = obj.lap_positional_encoding(g)
# fpath = os.getcwd() + '/data_lap_pe/coauthorphysics_lap_pe.npz'
# np.savez(fpath, lap_pe) 
