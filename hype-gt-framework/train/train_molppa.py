"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from ogb.graphproppred import Evaluator

def train_epoch(model, optimizer, device, data_loader, net_params):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    evaluator = Evaluator(name = "ogbg-ppa")
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)

    for iter, batch_graphs in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        # batch_x = batch_graphs.ndata['feat'].to(device)
        # batch_e = batch_graphs.edata['feat'].to(device)
        batch_x = batch_graphs.x.to(device)
        batch_e = batch_graphs.edge_attr.to(device)
        batch_labels = batch_graphs.y.to(device)
        optimizer.zero_grad()
    
        if net_params['lap_pos_enc'] is True:
            # batch_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            batch_pos_enc = batch_graphs.pos_enc.to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        else:
            # batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
            batch_pos_enc = batch_graphs.pos_enc.to(device)

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
    
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        targets = torch.cat([targets, batch_labels], 0)
        scores = torch.cat([scores, batch_scores], 0)

    scores = torch.argmax(scores, dim = 1, keepdim = True)
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_acc = evaluator.eval(input_dict)['acc']  
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, net_params):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    evaluator = Evaluator(name = "ogbg-ppa")
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)

    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            # batch_x = batch_graphs.ndata['feat'].to(device)
            # batch_e = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.x.to(device)
            batch_e = batch_graphs.edge_attr.to(device)
            batch_labels = batch_graphs.y.to(device)
            
            if net_params['lap_pos_enc'] is True:
                # batch_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                batch_pos_enc = batch_graphs.pos_enc.to(device)
            else:
                # batch_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
                batch_pos_enc = batch_graphs.pos_enc.to(device)
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()

            targets = torch.cat((targets, batch_labels), 0)
            scores = torch.cat((scores, batch_scores), 0)
        
        scores = torch.argmax(scores, dim = 1, keepdim = True)
        input_dict = {"y_true": targets, "y_pred": scores}
        epoch_test_acc = evaluator.eval(input_dict)['acc']  
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


