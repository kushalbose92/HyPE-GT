"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
from torch._C import dtype
import torch.nn as nn
import math

from ogb.graphproppred import Evaluator
from train.MetricWrapper import MetricWrapper


def train_epoch(model, optimizer, device, data_loader, epoch, batch_accumulation, net_params):
    model.train()
    evaluator = Evaluator(name = "ogbg-molpcba")

    epoch_loss = 0

    targets=torch.tensor([])
    scores=torch.tensor([])

    wrapped_loss_fun = MetricWrapper(metric=model.loss, target_nan_mask="ignore-flatten")

    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        # print(iter, torch.cuda.memory_allocated(0))
        batch_graphs = batch_graphs.to(device=device)
        batch_x = batch_graphs.ndata['feat']
        batch_e = batch_graphs.edata['feat']

        batch_targets = batch_targets.to(device)

        if net_params['lap_pos_enc'] is True:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        else:
            batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
        
        # batch_scores = model.forward(batch_graphs, batch_x, batch_e, None)
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)

        loss = wrapped_loss_fun(batch_scores, batch_targets)
        loss = loss / batch_accumulation
        loss.backward()

        # weights update
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()

        targets = torch.cat((targets, batch_targets.detach().cpu()), 0)
        scores = torch.cat((scores, batch_scores.detach().cpu()), 0)

    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_ap = evaluator.eval(input_dict)['ap']

    epoch_loss /= (iter + 1)

    return epoch_loss, epoch_train_ap, optimizer


def evaluate_network(model, device, data_loader, epoch, net_params):
    model.eval()
    evaluator = Evaluator(name = "ogbg-molpcba")

    epoch_test_loss = 0

    targets=torch.tensor([])
    scores=torch.tensor([])

    wrapped_loss_fun = MetricWrapper(metric=model.loss, target_nan_mask="ignore-flatten")

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device=device)
            batch_x = batch_graphs.ndata['feat']
            batch_e = batch_graphs.edata['feat']
            batch_targets = batch_targets.to(device=device)

            if net_params['lap_pos_enc'] is True:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            else:
                batch_lap_pos_enc = batch_graphs.ndata['rw_pos_enc'].to(device)
            
            # batch_scores = model.forward(batch_graphs, batch_x, batch_e, None)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)

            loss = wrapped_loss_fun(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            targets = torch.cat((targets, batch_targets.detach().cpu()), 0)
            scores = torch.cat((scores, batch_scores.detach().cpu()), 0)


    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_test_ap = evaluator.eval(input_dict)['ap']

    epoch_test_loss /= (iter + 1)

    return epoch_test_loss, epoch_test_ap

