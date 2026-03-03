import logging
from model import *
from utils import *
import torch
import math
import os
import csv
import pickle as pkl
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import argparse
import time
from sklearn.model_selection import KFold
import glob
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    average_precision_score
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos, batch_edges_neg,
                          batch_edge_weight_neg, batch_y_pos, batch_y_neg):
    device = next(classifier_model.parameters()).device
    x_pos = torch.tensor(batch_edges_pos, dtype=torch.long, device=device)
    x_neg = torch.tensor(batch_edges_neg, dtype=torch.long, device=device)
    w_pos = torch.tensor(batch_edge_weight_pos, device=device)
    w_neg = torch.tensor(batch_edge_weight_neg, device=device)

    if len(batch_y_pos) == 0:
        x = torch.cat([x_pos, x_neg])
        w = torch.cat([w_pos, w_neg])
        y = torch.cat([torch.ones((len(x_pos)), device=device), torch.zeros((len(x_neg)), device=device)])

        index = torch.randperm(len(x), device=device)
        x, y, w = x[index], y[index], w[index]


    pred = classifier_model(x, return_recon=False)
    pred = pred.squeeze(1)
    loss = loss_f(pred, y, weight=w)
    return pred, y, loss


def train_epoch(args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, train_set, metabolite_count,
                reaction_count, loss_f, training_data, optimizer, batch_size):
    net.train()
    classifier_model.train()

    edges_pos, edge_weight_pos, edges_neg, edge_weight_neg = training_data

    max_len = max(len(edge) for edge in edges_pos)
    edges_pos_padded_pos = [edge + [0] * (max_len - len(edge)) for edge in edges_pos]
    edges_pos_padded_neg = [edge + [0] * (max_len - len(edge)) for edge in edges_neg]
    edges_pos = torch.tensor(edges_pos_padded_pos, dtype=torch.long, device=device)
    edges_neg = torch.tensor(edges_pos_padded_neg, dtype=torch.long, device=device)

    index = torch.randperm(len(edges_pos)).numpy()
    edges_pos, edge_weight_pos = edges_pos[index], edge_weight_pos[index]
    edges_neg, edge_weight_neg = edges_neg[index], edge_weight_neg[index]

    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []
    y_pos = torch.tensor([])
    y_neg = torch.tensor([])

    batch_num = int(math.floor(len(edges_pos) / batch_size))

    for i in range(batch_num):
        for opt in optimizer:
            opt.zero_grad()

        batch_edges_pos = edges_pos[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_pos = edge_weight_pos[i * batch_size:(i + 1) * batch_size]
        batch_edges_neg = edges_neg[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_neg = edge_weight_neg[i * batch_size:(i + 1) * batch_size]

        batch_y_pos = ""

        X1, X2, Y_pos, Y_neg = net(m_emb, adj_matrix, hg_pos, hg_neg)

        if X1.dim() > 2:
            X1 = X1.squeeze()
        classifier_model.set_node_embedding(X1)

        if len(y_pos) > 0:
            batch_y_pos = y_pos[i * batch_size:(i + 1) * batch_size]
            batch_y_neg = y_neg[i * batch_size:(i + 1) * batch_size]
        else:
            batch_y_pos = []
            batch_y_neg = []

        pred, batch_y, loss = train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos,
                                                    batch_edges_neg, batch_edge_weight_neg, batch_y_pos, batch_y_neg)

        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)

        loss.backward()
        for opt in optimizer:
            opt.step()

        bce_total_loss = bce_total_loss + loss.item()

    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    auroc, auprc = roc_auc_cuda(y, pred)
    return bce_total_loss / batch_num, np.mean(acc_list), auroc, auprc


def eval_epoch(args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, loss_f, validation_data, batch_size):
    bce_total_loss = 0
    net.eval()
    classifier_model.eval()

    valid_edges_pos_set, valid_weight_pos_set, valid_edges_neg_set, valid_weight_neg_set = validation_data

    max_len = max(len(edge) for edge in valid_edges_pos_set)
    valid_edges_pos_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_pos_set]
    valid_edges_neg_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_neg_set]

    valid_edges_pos_set = torch.tensor(valid_edges_pos_padded, dtype=torch.long, device=device)
    valid_edges_neg_set = torch.tensor(valid_edges_neg_padded, dtype=torch.long, device=device)
    valid_weight_pos_set = torch.tensor(valid_weight_pos_set, device=device)
    valid_weight_neg_set = torch.tensor(valid_weight_neg_set, device=device)

    with torch.no_grad():
        index = torch.randperm(len(valid_edges_pos_set), device=device)
        valid_edges_pos_set = valid_edges_pos_set[index]

        pred_list, label_list, score_list = [], [], []

        num_batches = int(math.floor(len(valid_edges_pos_set) / batch_size))
        for i in range(num_batches):

            X1, X2, Y_pos, Y_neg = net(m_emb, adj_matrix, hg_pos, hg_neg)

            if X1.dim() > 2: X1 = X1.squeeze()
            classifier_model.set_node_embedding(X1)

            batch_x_pos = valid_edges_pos_set[i * batch_size: (i + 1) * batch_size]
            batch_w_pos = valid_weight_pos_set[i * batch_size: (i + 1) * batch_size]
            batch_x_neg = valid_edges_neg_set[i * batch_size: (i + 1) * batch_size]
            batch_w_neg = valid_weight_neg_set[i * batch_size: (i + 1) * batch_size]

            batch_x = torch.cat([batch_x_pos, batch_x_neg])
            batch_w = torch.cat([batch_w_pos, batch_w_neg])
            batch_y = torch.cat(
                [torch.ones(len(batch_x_pos), device=device), torch.zeros(len(batch_x_neg), device=device)])

            pred_batch = classifier_model(batch_x, return_recon=False)
            pred_batch = pred_batch.squeeze(1)
            loss = loss_f(pred_batch, batch_y, weight=batch_w)

            pred_list.append((pred_batch >= 0.5).float())
            label_list.append(batch_y)
            score_list.append(pred_batch)
            bce_total_loss += loss.item()

    pred = torch.cat(pred_list)
    label = torch.cat(label_list)
    scores = torch.cat(score_list)

    acc = accuracy(pred, label)
    auroc, auprc = roc_auc_cuda(label.cpu(), scores.cpu())

    y_np = label.cpu().numpy()
    pred_np = pred.cpu().numpy()
    precision = precision_score(y_np, pred_np)
    recall = recall_score(y_np, pred_np)
    f1 = f1_score(y_np, pred_np)
    mcc = matthews_corrcoef(y_np, pred_np)

    return {
        'bce_loss': bce_total_loss / (i + 1),
        'acc': acc, 'auroc': auroc, 'auprc': auprc,
        'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc
    }


def train(args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, train_set, valid_set, metabolite_count,
          reaction_count, loss_f, training_data, validation_data, optimizer, epochs, batch_size, fold, model_name):
    best_auroc = 0.0
    best_metrics = {
        'epoch': 0, 'valid_bce_loss': float('inf'), 'valid_acc': 0, 'valid_auroc': 0,
        'valid_auprc': 0, 'valid_precision': 0, 'valid_recall': 0, 'valid_f1': 0, 'valid_mcc': 0
    }


    epoch_iterator = tqdm(range(epochs), desc=f"Fold {fold}", unit="ep", ncols=120)

    for epoch_i in epoch_iterator:
        bce_loss, train_acc, auroc, auprc = train_epoch(
            args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, train_set, metabolite_count, reaction_count,
            loss_f, training_data, optimizer, batch_size)

        valid_results = eval_epoch(args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, loss_f,
                                   validation_data, batch_size)


        epoch_iterator.set_postfix(
            Loss=f"{valid_results['bce_loss']:.4f}",
            AUC=f"{valid_results['auroc']:.3f}",
            Acc=f"{100 * valid_results['acc']:.1f}"
        )

        if valid_results['auroc'] > best_auroc:
            best_auroc = valid_results['auroc']
            best_metrics = {
                'epoch': epoch_i,
                'valid_bce_loss': valid_results['bce_loss'],
                'valid_acc': valid_results['acc'],
                'valid_auroc': valid_results['auroc'],
                'valid_auprc': valid_results['auprc'],
                'valid_precision': valid_results['precision'],
                'valid_recall': valid_results['recall'],
                'valid_f1': valid_results['f1'],
                'valid_mcc': valid_results['mcc']
            }
            checkpoint = {
                'net': net.state_dict(),
                'classifier_model': classifier_model.state_dict(),
                'epoch': epoch_i,
                'metrics': valid_results
            }
            model_filename = f'trained_model_fold{fold}_{model_name}.pth'
            save_path = os.path.join(args.save_path, model_filename)
            torch.save(checkpoint, save_path)

    torch.cuda.empty_cache()
    return net, classifier_model, best_metrics