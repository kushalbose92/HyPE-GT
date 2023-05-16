import torch
import torch_geometric.nn
from torch_geometric.datasets import GNNBenchmarkDataset

trainset = GNNBenchmarkDataset(root='data/', name='MNIST', split='train')
valset = GNNBenchmarkDataset(root='data/', name='MNIST', split='val')
testset = GNNBenchmarkDataset(root='data/', name='MNIST', split='test')
print(len(trainset))
print(valset)
print(testset)