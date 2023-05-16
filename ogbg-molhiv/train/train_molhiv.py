"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from ogb.graphproppred import Evaluator

def train_epoch(model, optimizer, device, data_loader, net_params):
    model.train()
    evaluator = Evaluator(name = "ogbg-molhiv")
    
    epoch_loss = 0
    epoch_train_auc = 0

    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()  
        
        if net_params['lap_pos_enc'] is True:
            batch_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        else:
            batch_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            
        targets = torch.cat((targets, batch_targets), 0)
        scores = torch.cat((scores, batch_scores), 0)
        
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        
    
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_auc = evaluator.eval(input_dict)['rocauc']  

    epoch_loss /= (iter + 1)
    
    return epoch_loss, epoch_train_auc, optimizer

def evaluate_network(model, device, data_loader, net_params):
    model.eval()
    evaluator = Evaluator(name = "ogbg-molhiv")
    
    epoch_test_loss = 0
    epoch_test_auc = 0
    
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            
            if net_params['lap_pos_enc'] is True:
                batch_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            else:
                batch_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
                
            targets = torch.cat((targets, batch_targets), 0)
            scores = torch.cat((scores, batch_scores), 0)         
            
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_test_auc = evaluator.eval(input_dict)['rocauc']
            
    epoch_test_loss /= (iter + 1)

    return epoch_test_loss, epoch_test_auc

