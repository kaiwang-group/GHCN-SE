import logging
import torch
import os
import csv
import pickle as pkl
import numpy as np
from dhg.random import set_seed
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import KFold
import glob
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import torch.nn.functional as F
import time
import warnings
from model import *
from utils import *
from train import *
import matplotlib as mpl

mpl.use("Agg")
import multiprocessing

cpu_num = multiprocessing.cpu_count()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")

def setup_logging(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='train Multi-HGNN for missing reaction prediction.')
    parser.add_argument('--top', default=100, type=int)
    parser.add_argument('--external_epochs', default=1, type=int)
    parser.add_argument('--internal_epochs', default=500, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--in_dim', default=512, type=int)
    parser.add_argument('--h_dim', default=256, type=int)
    parser.add_argument('--out_dim', default=64, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--num_dgnn', default=2, type=int)
    parser.add_argument('--num_hgnn', default=1, type=int)
    parser.add_argument('--data', type=str, default='model')
    parser.add_argument('--dimensions', type=int, default=64)
    parser.add_argument('-d', '--diag', type=str, default='True')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()

    batch_size = 96
    bottle_neck = args.dimensions
    train_type = 'hyper'

    universe_pool = cobra.io.read_sbml_model('./data/pools/bigg_universe.xml')
    universe_pool_copy = universe_pool.copy()
    rxn_pool_df = cobra.util.array.create_stoichiometric_matrix(
        universe_pool_copy, array_type='DataFrame'
    )
    bigg_metabolite = set(rxn_pool_df.index)
    bigg_reaction = set(rxn_pool_df.columns)

    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))
    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        args.save_path = os.path.join('./checkpoints/', args.data, model_name)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        logger = setup_logging(args)
        print(f'Current model ID: {model_name}')

        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        read_xml(xml_content, rxn_pool_df, bigg_metabolite, bigg_reaction)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pos = os.path.join(current_dir, 'incidence_matrix_pos.csv')
        file_path_neg = os.path.join(current_dir, 'incidence_matrix_neg.csv')
        file_path_bigg = os.path.join(current_dir, 'incidence_matrix_bigg.csv')

        df_pos = pd.read_csv(file_path_pos)
        df_neg = pd.read_csv(file_path_neg)
        df_bigg = pd.read_csv(file_path_bigg)

        reaction_count_total = len(reaction)

        pos_dict = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_dict()
        neg_dict = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_dict()
        edges_bigg = df_bigg.groupby('hyperedge_id')['node_id'].apply(list).to_list()

        edges_pos = [pos_dict.get(i, []) for i in range(reaction_count_total)]
        edges_neg = [neg_dict.get(i, []) for i in range(reaction_count_total)]
        edges_bigg = [edges_bigg.get(i, []) for i in range(reaction_count_total)]

        edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
        edges_weight_neg = np.ones(len(edges_neg), dtype='float32')
        edges_weight_bigg = np.ones(len(edges_bigg), dtype='float32')

        reaction = pd.read_csv('reaction.csv')
        reaction_to_hyperedge = dict(zip(reaction.index, reaction['hyperedge_id']))

        fold_assignments = []
        adj_matrix, hg_pos, hg_neg, initial_features, reaction_count, metabolite_count = process_data()

        assert reaction_count == reaction_count_total
        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=0)
        m_emb = initial_features

        for fold, (train_set, valid_set) in enumerate(kf.split(reaction_count)):

         
            min_data_size = min(len(train_set), len(valid_set))
            candidate_batch_sizes = [96, 64, 32]
            current_batch_size = 16
            for size in candidate_batch_sizes:
                if min_data_size >= size:
                    current_batch_size = size
                    break

            train_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in train_set]
            valid_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in valid_set]

            train_edges_pos_set = [edges_pos[i] for i in train_hyperedge_ids_set]
            valid_edges_pos_set = [edges_pos[i] for i in valid_hyperedge_ids_set]
            train_edges_neg_set = [edges_neg[i] for i in train_hyperedge_ids_set]
            valid_edges_neg_set = [edges_pos[i] for i in valid_hyperedge_ids_set]

            train_weight_pos_set = edges_weight_pos[train_hyperedge_ids_set]
            valid_weight_pos_set = edges_weight_pos[valid_hyperedge_ids_set]
            train_weight_neg_set = edges_weight_neg[train_hyperedge_ids_set]
            valid_weight_neg_set = edges_weight_bigg[valid_hyperedge_ids_set]

            net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_dgnn, args.num_hgnn,
                        num_nodes=metabolite_count, use_bn=True)

            classifier_model = Classifier(
                n_head=8, d_model=args.dimensions, d_k=16, d_v=16, node_embedding=None,
                metabolite_count=metabolite_count, diag_mask=args.diag, bottle_neck=bottle_neck,
                device=device
            ).to(device)

            params_list = list(set(list(classifier_model.parameters()) + list(net.parameters())))
            optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.lr)

            m_emb = m_emb.to(device)
            adj_matrix = adj_matrix.to(device)
            hg_pos = hg_pos.to(device)
            hg_neg = hg_neg.to(device)

            net = net.to(device)
            classifier_model = classifier_model.to(device)
            train_set_tensor = torch.tensor(train_set).long()
            valid_set_tensor = torch.tensor(train_set).long()

            loss_f = F.binary_cross_entropy

            best_fold_scores = None

            for epoch in range(args.external_epochs):
                net, classifier_model, fold_scores = train(
                    args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg,
                    train_set_tensor, valid_set_tensor, metabolite_count, reaction_count, loss_f,
                    training_data=(train_edges_pos_set, train_weight_pos_set, train_edges_neg_set, train_weight_neg_set),
                    validation_data=(valid_edges_pos_set, valid_weight_pos_set, valid_edges_neg_set, valid_weight_neg_set),
                    optimizer=[optimizer],
                    epochs=args.internal_epochs,
                    batch_size=current_batch_size,
                    fold=fold,
                    model_name=model_name)
                
                best_fold_scores = fold_scores
            

            fold_df = pd.DataFrame({'reaction_id': valid_set, 'score': best_fold_scores})
            fold_csv_path = os.path.join(args.save_path, f'reaction_scores_fold_{fold}.csv')
            fold_df.to_csv(fold_csv_path, index=False)
            print(f"Fold {fold} scores saved to {fold_csv_path}")
	
        