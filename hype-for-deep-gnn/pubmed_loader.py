import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random
import dgl 
from scipy import sparse as sp
import numpy as np 
import os

class Pubmed():

    def __init__(self):

        dataset = Planetoid(root='data/Planetoid', name='PubMed', transform = NormalizeFeatures())
        data = dataset[0]

        self.name = "Pubmed"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.train_mask_sum = data.train_mask.sum()
        self.train_label_rate = (int(data.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.data_contains_self_loops = data.contains_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

    def adj_list_generation(self, edge_index):

        adj_list = [[] for n in range(self.num_nodes)]
        src_list = edge_index[0]
        dest_list = edge_index[1]

        for n in range(self.num_edges):

            adj_list[int(src_list[n])].append(int(dest_list[n]))

        return adj_list

    # path generation
    def path_generator(self, adj_list, length, path_per_node):

        path_list = torch.zeros(path_per_node * self.num_nodes, length+1, dtype = torch.long)
        path_count = 0

        for n_idx in range(self.num_nodes):

            head_node = n_idx
            degree = len(adj_list[head_node])
            neighs = random.sample(adj_list[head_node], path_per_node if degree > path_per_node else degree)

            for _ in range(len(neighs)):

                path_list[path_count][0] = head_node
                curr_node = head_node

                for l in range(1, length+1):

                    next_node = random.sample(adj_list[curr_node], 1)[0]
                    path_list[path_count][l] = next_node
                    curr_node = next_node

                path_count += 1

        print("Path count ", path_count)
        path_list = path_list[:path_count]

        return path_list


    def data_visualize(self, img_name):

        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(img_name + ".png")
        plt.clf()


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


# pubmed = Pubmed()
# edge_index = pubmed.edge_index
# g = dgl.graph((edge_index[0], edge_index[1]))
# lap_pe = pubmed.lap_positional_encoding(g)
# fpath = os.getcwd() + '/data_lap_pe/pubmed_lap_pe.npz'
# np.savez(fpath, lap_pe) 