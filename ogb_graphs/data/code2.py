import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib

from ogb.graphproppred import PygGraphPropPredDataset, collate_dgl

from tqdm.std import tqdm

### for data transform
from utils.code2_utils import augment_edge, encode_y_to_arr, decode_arr_to_seq



def laplacian_positional_encoding(graph, max_freqs):
    g = dgl.graph((graph.edge_index[0], graph.edge_index[1]))

    # Laplacian
    n = g.number_of_nodes()
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt="csr")
    # A = g.adjacency_matrix(transpose, scipy_fmt="csr").
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        # g.ndata['lap_pos_enc'] = F.pad(EigVecs, (0, max_freqs - n), value=float(0))
        graph.lap_pos_enc = F.pad(EigVecs, (0, max_freqs - n), value=float(0))
    else:
        # g.ndata['lap_pos_enc'] = EigVecs
        graph.lap_pos_enc = EigVecs

    # Save eigenvalues and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(
        EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float(0)).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    # Save EigVals node features
    # g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(), 1).unsqueeze(2)

    return graph


# random walk positional encodings
def rand_walk_positional_encoding(graph, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
    """
    g = dgl.graph((graph.edge_index[0], graph.edge_index[1]))
    n = g.number_of_nodes()

    # Geometric diffusion features with Random Walk
    # A = g.adjacency_matrix(scipy_fmt="csr")
    A = g.adj_external(scipy_fmt="csr")
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
    graph.rw_pos_enc = PE  
    
    return graph


def make_full_graph(graph):
    g, label = graph

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Copy over the node feature data and laplace eigvals/eigvecs
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Initalize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over, and identify real edges!
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)  # This indicates real edges

    return full_g, label


class Code2Dataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Code2 dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        dataset = PygGraphPropPredDataset(name='ogbg-code2')
        self.split_idx = dataset.get_idx_split()

        self.dataset = dataset
        
        # print("labels  ", self.dataset.data.y)

        # seq_len_list = np.array([len(seq) for seq in dataset.y])
        # # print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)))

        # split_idx = dataset.get_idx_split()

        # vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
        # dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(dataset, vocab2idx, max_seq_len)])

        # self.train = dataset[split_idx["train"]]
        # self.val = dataset[split_idx["valid"]]
        # self.test = dataset[split_idx["test"]]

        # print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        # print("[I] Finished loading.")
        # print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        return batched_graph, labels


    # def _make_full_graph(self):
    #     self.train = [make_full_graph(graph) for graph in tqdm(self.train)]
    #     self.val = [make_full_graph(graph) for graph in tqdm(self.val)]
    #     self.test = [make_full_graph(graph) for graph in tqdm(self.test)]


    def _laplacian_positional_encoding(self, max_freqs, train, val, test):
        train = [laplacian_positional_encoding(graph, max_freqs) for graph in tqdm(train)]
        val = [laplacian_positional_encoding(graph, max_freqs) for graph in tqdm(val)]
        test = [laplacian_positional_encoding(graph, max_freqs) for graph in tqdm(test)]
        return train, val, test


    def rand_walk_positional_encoding(self, max_freqs, train, val, test):
        train = [rand_walk_positional_encoding(graph, max_freqs) for graph in tqdm(train)]
        val = [rand_walk_positional_encoding(graph, max_freqs) for graph in tqdm(val)]
        test = [rand_walk_positional_encoding(graph, max_freqs) for graph in tqdm(test)]
        return train, val, test

    