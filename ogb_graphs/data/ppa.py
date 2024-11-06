import time
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_scatter import scatter
from functools import partial
import torch_geometric.utils as utils

from ogb.graphproppred import DglGraphPropPredDataset, PygGraphPropPredDataset, Evaluator

from scipy import sparse as sp
import numpy as np
import networkx as nx
# from tqdm import tqdm
from tqdm.std import tqdm


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data

def lap_positional_encoding(graph, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    use_edge_attr = False

    edge_attr = graph.edge_attr if use_edge_attr else None
    edge_index, edge_attr = utils.get_laplacian(
        graph.edge_index, edge_attr, normalization='sym',
        num_nodes=graph.num_nodes)
    L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()


# random walk positional encodings
def rw_positional_encoding(graph, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
    """
    W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
    W = W0
    vector = torch.zeros((graph.num_nodes, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    return vector.float()


class OGBPPADataset(Dataset):
    def __init__(self, name):

        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name.lower()
        
        transform = partial(extract_node_feature, reduce='add')
        self.dataset = PygGraphPropPredDataset(name=self.name, transform = transform)
        split_idx = self.dataset.get_idx_split()

        self.train = self.dataset[split_idx["train"]]
        self.val = self.dataset[split_idx["valid"]]
        self.test = self.dataset[split_idx["test"]]
        
        print('train, val, test sizes :', len(self.train), len(self.val), len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def _add_laplacian_positional_encodings(self, pos_enc_dim, dataset):
        dataset.pos_enc_list = []
        for g in tqdm(dataset):
            pos_enc = lap_positional_encoding(g, pos_enc_dim)
            dataset.pos_enc_list.append(pos_enc)
        return dataset

    def _add_rand_walk_positional_encodings(self, pos_enc_dim, dataset):
        dataset.pos_enc_list = []
        for g in tqdm(dataset):
            pos_enc = rw_positional_encoding(g, pos_enc_dim)
            dataset.pos_enc_list.append(pos_enc)
        return dataset


class GraphDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pos_enc_list = self.dataset.pos_enc_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data.pos_enc = None
        data.pos_enc = self.pos_enc_list[index]
        return data

 
        

    
