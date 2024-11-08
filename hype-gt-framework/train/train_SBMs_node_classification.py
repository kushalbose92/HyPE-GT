"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

def train_epoch(model, optimizer, device, data_loader, net_params):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
    
        if net_params['lap_pos_enc'] is True:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        else:
            batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
    
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, net_params):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            
            if net_params['lap_pos_enc'] is True:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            else:
                batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
        
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


