import argparse
import dgl
import torch 
import torch.nn.functional as F
from deep_gnn import *
from cora_loader import Cora 
from citeseer_loader import Citeseer
from pubmed_loader import Pubmed
from coauthor_cs_loader import CoauthorCS
from coauthor_physics_loader import CoauthorPhysics
from amazon_photo_loader import AmazonPhoto
from amazon_computers_loader import AmazonComputers
# from dgl import LaplacianPE
import os
import sys
import random
import numpy as np 
from utils_1 import lap_positional_encoding, rand_walk_positional_encoding, loss_fn, adj_normalization, visualize

import warnings
warnings.filterwarnings("ignore")


def argument_parser():
    
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--lr', help = 'learning rate', default = 0.2, type = float)
    parser.add_argument('--seed', help = 'Random seed', default = 100, type = int)
    parser.add_argument('--pos_enc_dim', help = 'hidden dimension for positional features', default = 16, type = int)
    parser.add_argument('--dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--pe_layers', help = 'number of layers', default = 2, type = int)
    parser.add_argument('--gcn_layers', help = 'number of gcn layers', default = 2, type = int)
    parser.add_argument('--manifold', help = 'name of the manifold', default = 'Hyperboloid', type = str)
    parser.add_argument('--pe_init',help = 'inital positional encodings',default= 'LapPE', type=str)
    parser.add_argument('--model', help = 'name of the model', default = 'HGCN', type = None)
    parser.add_argument('--gnn_model', help= 'name of the gnn model', default= 'gcn', type=str)
    parser.add_argument('--act', help = 'activation function', default = 'relu', type = str)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 200, type = int)
    parser.add_argument('--test_iter', help = 'number of test iteration', default = 10, type = int)
    parser.add_argument('--use_saved_model', help = 'use saved model in directory', default = False, type = None)
    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--w_decay', help = 'Weight decay for the optimizer', default = 0.0005, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)
    parser.add_argument('--c', help = 'curvature of the manifold', default = 1.0, type = float)
    parser.add_argument('--run_base', help = 'Train in Euclidean Space', default = False, type = None)
    parser.add_argument('--pe_category', help = 'Category of the Positional Encodings', default= 1, type = int)

    return parser


parsed_args = argument_parser().parse_args()
c = parsed_args.c
act = parsed_args.act 
pe_layers = parsed_args.pe_layers
# manifold = parsed_args.manifold
# pe_init = parsed_args.pe_init
# model = parsed_args.model 
gnn_model = parsed_args.gnn_model
pos_enc_dim = parsed_args.pos_enc_dim
dim = parsed_args.dim 
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
w_decay = parsed_args.w_decay
dropout = parsed_args.dropout 
gcn_layers = parsed_args.gcn_layers
device = 'cuda:' + str(parsed_args.device)
dataset = parsed_args.dataset
seed = parsed_args.seed
# run_base = parsed_args.run_base
pe_category = parsed_args.pe_category

pe_init, manifold, model, run_base = None, None, None, False 

if pe_category == 1:
    pe_init = 'LapPE'
    manifold = 'Hyperboloid'
    model = 'HGCN'
elif pe_category == 2:
    pe_init = 'LapPE'
    manifold = 'Hyperboloid'
    model = 'HNN'
elif pe_category == 3:
    pe_init = 'LapPE'
    manifold = 'PoincareBall'
    model = 'HGCN'
elif pe_category == 4:
    pe_init = 'LapPE'
    manifold = 'PoincareBall'
    model = 'HNN'
elif pe_category == 5:
    pe_init = 'RWPE'
    manifold = 'Hyperboloid'
    model = 'HGCN'
elif pe_category == 6:
    pe_init = 'RWPE'
    manifold = 'Hyperboloid'
    model = 'HNN'
elif pe_category == 7:
    pe_init = 'RWPE'
    manifold = 'PoincareBall'
    model = 'HGCN'
elif pe_category == 8:
    pe_init = 'RWPE'
    manifold = 'PoincareBall'
    model = 'HNN'
elif pe_category == 0:
    run_base = True
else:
    print("wrong choice entered\n")

def apply_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if device == 'cuda:0':
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

apply_seed(seed)

print(dataset)
if run_base == 'False':
    print("{} || {} || {} || {} || {} || {}".format(model, manifold, gnn_model, pe_init, gcn_layers, pe_layers))

if dataset == 'cora':
    data_obj = Cora()
elif dataset == 'citeseer':
    data_obj = Citeseer()
elif dataset == 'pubmed':
    data_obj = Pubmed()
elif dataset == 'coauthorcs':
    data_obj = CoauthorCS()
elif dataset == 'amazonphoto':
    data_obj = AmazonPhoto()
elif dataset == 'amazoncomputers':
    data_obj = AmazonComputers()
elif dataset == 'coauthorphysics':
    data_obj = CoauthorPhysics()
else:
    print("Invalid dataset name")


# generating Laplacian positional embeddings
# transform = LaplacianPE(k = pos_enc_dim, feat_name = 'eigvec')
# src = data_obj.edge_index[0].numpy()
# tgt = data_obj.edge_index[1].numpy()
# g = dgl.graph((src, tgt))
# g_trans = transform(g)
# laplacian_pe = g_trans.ndata['eigvec'].to(device)
# print("laplacian pe ", laplacian_pe.shape)

src = data_obj.edge_index[0].numpy()
tgt = data_obj.edge_index[1].numpy()
g = dgl.graph((src, tgt))

if run_base == False:
    if pe_init == 'LapPE':
        fpath = os.getcwd() + '/data_lap_pe/' + dataset + '_lap_pe.npz'
        f = np.load(fpath)
        pe_init = f['arr_0']
        pe_init = torch.from_numpy(pe_init[:,1:pos_enc_dim+1]).float() 
        g.ndata['pos_enc'] = pe_init
        print("Laplacian PE generated")
        # pe_init = lap_positional_encoding(g, pos_enc_dim)
        # print(pe_init.shape)
    else:
        pe_init = rand_walk_positional_encoding(g, pos_enc_dim)
        print("Rand Walk PE generated")
    pe_init = pe_init.to(device)
else:
    print("No init PE")
# laplacian_pe = lap_positional_encoding(g, pos_enc_dim)


if gnn_model == 'gcnii':
    model = GCNII(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
elif gnn_model == 'gcn':
    model = GCN(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
elif gnn_model == 'gat':
    model = GAT(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
elif gnn_model == 'sage':
    model = SAGE(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
elif gnn_model == 'gin':
    model = GIN(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
elif gnn_model == 'jknet':
    model = JKNet(data_obj, gcn_layers, c, act, pe_layers, manifold, model, pos_enc_dim, dim, dropout, run_base, device)
    
model.to(device)

opti = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = w_decay)

data_obj.node_features = data_obj.node_features.to(device)
data_obj.edge_index = data_obj.edge_index.to(device)

model_path = os.getcwd() + "/saved_models/" + dataset + "_" + str(gcn_layers) 


def test(model, data_obj, adj, edge_index, lap_pe):

    model.eval()
    correct = 0
    emb, pred = model(data_obj.node_features, adj, edge_index, lap_pe)
    pred = pred.argmax(dim = 1)
    label = data_obj.node_labels.to(device)
    pred = pred[data_obj.test_mask].to(device)
    label = label[data_obj.test_mask]
    correct = pred.eq(label).sum().item()
    acc = correct / int(data_obj.test_mask.sum())

    return acc

adj = adj_normalization(data_obj).to(device)

best_test_acc = 0.0
best_epoch = 0
model.train()
for epoch in range(train_iter):

    model.train()
    opti.zero_grad()
    emb, pred = model(data_obj.node_features, adj, data_obj.edge_index, pe_init)
    label = data_obj.node_labels
    pred = pred[data_obj.train_mask]
    label = label[data_obj.train_mask]
    pred = pred.to(device)
    label = label.to(device)
    loss = loss_fn(pred, label)
    loss.backward()
    opti.step()

    test_acc = test(model, data_obj, adj, data_obj.edge_index, pe_init)
    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write(f"Epoch: {epoch+1:03d} || Loss: {loss:.4f} || Test Acc: {test_acc:.4f}")

    if test_acc > best_test_acc:
        torch.save(model.state_dict(), model_path) 
        best_test_acc = test_acc
        best_epoch = epoch
        
print("Best model saved in ", best_epoch, " with test accuracy ", best_test_acc)


print("\n*********Model Evaluation*********\n")

test_acc_list = []
model.load_state_dict(torch.load(model_path))
model.eval()

for i in range(test_iter):
    test_acc = test(model, data_obj, adj, data_obj.edge_index, pe_init)
    print(f"Test Accuracy: {test_acc:.4f}")
    test_acc_list.append(test_acc)

print(f"Test statistics: {np.mean(test_acc_list):.4f} || {np.std(test_acc_list): .4f}")

# embedding visualization
# emb, out = model(data_obj.node_features, adj, data_obj.edge_index, pe_init)
# visualize(emb, data_obj.node_labels, data_obj.name, gcn_layers)