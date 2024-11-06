"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
from torch._C import dtype
import torch.nn as nn
import math
import numpy as np

from ogb.graphproppred import Evaluator
from train.MetricWrapper import MetricWrapper


def train_epoch(model, optimizer, device, data_loader, epoch, batch_accumulation, net_params, arr_to_seq):
    model.train()
    evaluator = Evaluator(name = "ogbg-code2")

    epoch_loss = 0.0
    seq_ref_list = []
    seq_pred_list = []

    multicls_criterion = nn.CrossEntropyLoss()

    for iter, batch_graphs in enumerate(data_loader):
        
        batch_graphs = batch_graphs.to(device=device)
        batch_x = batch_graphs.x
        batch_e = batch_graphs.edge_attr

        if net_params['lap_pos_enc'] is True:
            # batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            batch_lap_pos_enc = batch_graphs.lap_pos_enc.to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        else:
            # batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
            batch_pos_enc = batch_graphs.rw_pos_enc.to(device)
        
        # pred_list = model.forward(batch_graphs, batch_x, batch_e, None)
        pred_list = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)

        loss = 0.0
        for i in range(len(pred_list)):
            loss += multicls_criterion(pred_list[i].to(torch.float32), batch_graphs.y_arr[:,i])
        loss = loss / len(pred_list)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # weights update
        # if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(data_loader)):
        #     optimizer.step()
        #     optimizer.zero_grad()

        mat = []
        for i in range(len(pred_list)):
            mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
        mat = torch.cat(mat, dim = 1)
        seq_pred = [arr_to_seq(arr) for arr in mat]

        # PyG >= 1.5.0
        seq_ref = [batch_graphs.y[i] for i in range(len(batch_graphs.y))]

        seq_ref_list.extend(seq_ref)
        seq_pred_list.extend(seq_pred)

        epoch_loss += loss.detach().item()

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    epoch_train_f1score = evaluator.eval(input_dict)['F1']

    epoch_loss /= (iter + 1)

    return epoch_loss, epoch_train_f1score, optimizer


def evaluate_network(model, device, data_loader, epoch, net_params, arr_to_seq):
    model.eval()
    evaluator = Evaluator(name = "ogbg-code2")

    epoch_test_loss = 0

    seq_ref_list = []
    seq_pred_list = []

    multicls_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device=device)
            batch_x = batch_graphs.x
            batch_e = batch_graphs.edge_attr

            if net_params['lap_pos_enc'] is True:
                # batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                batch_pos_enc = batch_graphs.lap_pos_enc.to(device)
            else:
                # batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
                batch_pos_enc = batch_graphs.rw_pos_enc.to(device)
            
            # pred_list = model.forward(batch_graphs, batch_x, batch_e, None)
            pred_list = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)

            loss = 0.0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch_graphs.y_arr[:,i])
            epoch_test_loss = loss / len(pred_list)

            epoch_test_loss += loss.detach().item()

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG >= 1.5.0
            seq_ref = [batch_graphs.y[i] for i in range(len(batch_graphs.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    epoch_test_f1score = evaluator.eval(input_dict)['F1']

    epoch_test_loss /= (iter + 1)

    return epoch_test_loss.item(), epoch_test_f1score

