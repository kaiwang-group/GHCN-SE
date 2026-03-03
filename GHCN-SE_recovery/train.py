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
        y = torch.cat([torch.ones((len(x_pos)), device=device), torch.zeros((len(x_neg)), device='cpu')])

        index = torch.randperm(len(x), device=device)
        x, y, w = x[index], y[index], w[index]

    pred = classifier_model(x, return_recon=False)
    pred = pred.squeeze(1)
    loss = loss_f(pred, y, weight=w_pos)
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
    edges_neg, edge_weight_neg = edges_neg[index], edge_weight_neg

    batch_num = int(math.floor(len(edges_pos) / batch_size))

    for i in range(batch_num):
        for opt in optimizer:
            opt.zero_grad()

        batch_edges_pos = edges_pos[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_pos = edge_weight_pos[i * batch_size:(i + 1) * batch_size]
        batch_edges_neg = edges_neg[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight_neg = edge_weight_neg[i * batch_size:(i + 1) * batch_size]

        batch_y_pos = ""

        X1, X2, Y_pos, Y_neg = net(m_emb, adj_matrix, hg_pos, hg_pos)

        if X1.dim() > 2:
            X1 = X1.squeeze()
        classifier_model.set_node_embedding(X1)

        if len(torch.tensor([])) > 0:
            batch_y_pos = [] 
            batch_y_neg = []
        else:
            batch_y_pos = []
            batch_y_neg = []

        pred, batch_y, loss = train_batch_hyperedge(classifier_model, loss_f, batch_edges_pos, batch_edge_weight_pos,
                                                    batch_edges_neg, batch_edge_weight_neg, batch_y_pos, batch_y_neg)

        loss.backward()
        for opt in optimizer:
            opt.step()

    return

def get_scores_epoch(args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg, validation_data, batch_size):
    net.eval()
    classifier_model.eval()
    
    valid_edges_pos_set, valid_weight_pos_set, _, _ = validation_data

    max_len = max(len(edge) for edge in valid_edges_pos_set)
    valid_edges_pos_padded = [edge + [0] * (max_len - len(edge)) for edge in valid_edges_pos_set]

    valid_edges_pos_set = torch.tensor(valid_edges_pos_padded, dtype=torch.float32, device=device)
    
    all_scores = []
    
    with torch.no_grad():
        num_batches = int(math.ceil(len(valid_edges_pos_set) / batch_size))
        for i in range(num_batches):

            X1, X2, Y_pos, Y_neg = net(m_emb, adj_matrix, hg_pos, hg_neg)

            if X1.dim() > 2: X1 = X1.squeeze()
            classifier_model.set_node_embedding(X1)

            batch_x_pos = valid_edges_pos_set[i * batch_size: (i + 1) * batch_size]
            
            pred_batch = classifier_model(batch_x_pos, return_recon=False)
            pred_batch = pred_batch.squeeze(1)
            
            all_scores.append(pred_batch.cpu().numpy())

    return np.concatenate(all_scores)

def train(args, net, classifier_model, m_emb, g, hg_pos, hg_neg, train_set, valid_set, metabolite_count,
          reaction_count, loss_f, training_data, validation_data, optimizer, epochs, batch_size, fold, model_name):
    
    final_scores = None

    for epoch_i in range(epochs):
        train_epoch(
            args, net, classifier_model, m_emb, g, hg_pos, hg_neg, train_set, metabolite_count, reaction_count,
            loss_f, training_data, optimizer, batch_size)

        scores = get_scores_epoch(args, net, classifier_model, m_emb, g, hg_pos, hg_neg,
                                          validation_data, batch_size)
        final_scores = scores

    return net, classifier_model, final_scores