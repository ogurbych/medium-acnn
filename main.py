import torch
import torch.nn as nn

from dgllife.utils.eval import Meter

def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ', {} {:.4f}'.format(metric, score)
    return msg

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter(args['train_mean'], args['train_std'])
    epoch_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        indices, ligand_mols, protein_mols, bg, labels = batch_data
        labels, bg = labels.to(args['device']), bg.to(args['device'])
        prediction = model(bg)
        loss = loss_criterion(prediction, (labels - args['train_mean']) / args['train_std'])
        epoch_loss += loss.data.item() * len(indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    avg_loss = epoch_loss / len(data_loader.dataset)
    total_scores = {metric: train_meter.compute_metric(metric, 'mean')
                    for metric in args['metrics']}
    msg = 'epoch {:d}/{:d}, training | loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], avg_loss)
    msg = update_msg_from_scores(msg, total_scores)
    print(msg)

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter(args['train_mean'], args['train_std'])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            indices, ligand_mols, protein_mols, bg, labels = batch_data
            labels, bg = labels.to(args['device']), bg.to(args['device'])
            prediction = model(bg)
            eval_meter.update(prediction, labels)
    total_scores = {metric: eval_meter.compute_metric(metric, 'mean')
                    for metric in args['metrics']}
    return total_scores