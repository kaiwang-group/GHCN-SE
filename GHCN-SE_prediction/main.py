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
    parser.add_argument('--external_epochs', default=1, type=int, help='maximum training external_epochs')
    parser.add_argument('--internal_epochs', default=500, type=int, help='maximum training epochs per fold')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--in_dim', default=512, type=int, help='dim of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='dim of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='dim of output embedding')
    parser.add_argument('--cuda', default=0, type=int, help='gpu index')
    parser.add_argument('--k_fold', default=5, type=int, help='k-fold cross validation')
    parser.add_argument('--num_gcn', default=2, type=int, help='the number of layers in the sparse gcn')
    parser.add_argument('--num_hg', default=1, type=int, help='the number of layers in the hypergraph neural network')
    parser.add_argument('--data', type=str, default='model')
    parser.add_argument('--dimensions', type=int, default=64, help='Number of dimensions. Default is 64.')
    parser.add_argument('-d', '--diag', type=str, default='True', help='Use the diag mask or not')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()

    batch_size = 96
    bottle_neck = args.dimensions
    pair_ratio = 0.9
    train_type = 'hyper'

    metric_collector = {
        'valid_bce_loss': [], 'valid_acc': [], 'valid_auroc': [], 'valid_auprc': [],
        'valid_precision': [], 'valid_recall': [], 'valid_f1': [], 'valid_mcc': []
    }

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
        read_xml(xml_content)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path_pos = os.path.join(current_dir, 'incidence_matrix_pos.csv')
        file_path_neg = os.path.join(current_dir, 'incidence_matrix_neg.csv')

        df_pos = pd.read_csv(file_path_pos)
        df_neg = pd.read_csv(file_path_neg)
        reaction = pd.read_csv('reaction.csv')

        reaction_count_total = len(reaction)


        pos_dict = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_dict()
        neg_dict = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_dict()


        edges_pos = [pos_dict.get(i, []) for i in range(reaction_count_total)]
        edges_neg = [neg_dict.get(i, []) for i in range(reaction_count_total)]


        edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
        edges_weight_neg = np.ones(len(edges_neg), dtype='float32')

        reaction_to_hyperedge = dict(zip(reaction.index, reaction['hyperedge_id']))

        fold_assignments = []

        adj_matrix, hg_pos, hg_neg, initial_features, reaction_count, metabolite_count = process_data()


        assert reaction_count == reaction_count_total, "Error: Mismatch in reaction counts!"

        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=0)
        m_emb = initial_features
        results = []

        all_folds_best_metrics = []

        for metric in metric_collector.values():
            metric.clear()


        for fold, (train_set, valid_set) in enumerate(kf.split(np.arange(reaction_count))):
            print(f"\n{'=' * 100}")


            min_data_size = min(len(train_set), len(valid_set))


            candidate_batch_sizes = [96, 64, 32]
            current_batch_size = 16

            for size in candidate_batch_sizes:
                if min_data_size >= size:
                    current_batch_size = size
                    break

            print(f"batch_size = {current_batch_size}")

            for idx in valid_set:
                bigg_id = reaction.iloc[idx]['bigg_id']
                fold_assignments.append({'bigg_id': bigg_id, 'fold': fold})


            train_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in train_set]
            valid_hyperedge_ids_set = [reaction_to_hyperedge[i] for i in valid_set]


            train_edges_pos_set = [edges_pos[i] for i in train_hyperedge_ids_set]
            valid_edges_pos_set = [edges_pos[i] for i in valid_hyperedge_ids_set]
            train_edges_neg_set = [edges_neg[i] for i in train_hyperedge_ids_set]
            valid_edges_neg_set = [edges_neg[i] for i in valid_hyperedge_ids_set]

            train_weight_pos_set = edges_weight_pos[train_hyperedge_ids_set]
            valid_weight_pos_set = edges_weight_pos[valid_hyperedge_ids_set]
            train_weight_neg_set = edges_weight_neg[train_hyperedge_ids_set]
            valid_weight_neg_set = edges_weight_neg[valid_hyperedge_ids_set]

            net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_gcn, args.num_hg,
                        num_nodes=metabolite_count, use_bn=True)


            classifier_model = Classifier(
                n_head=8, d_model=args.dimensions, d_k=16, d_v=16, node_embedding=None,
                metabolite_count=metabolite_count, diag_mask=args.diag, bottle_neck=bottle_neck,
                device=device
            ).to(device)

            params_list = list(set(list(classifier_model.parameters()) + list(net.parameters())))
            optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.wd)

            m_emb = m_emb.to(device)
            adj_matrix = adj_matrix.to(device)


            hg_pos = hg_pos.to(device)
            hg_neg = hg_neg.to(device)

            net = net.to(device)
            classifier_model = classifier_model.to(device)
            train_set = torch.tensor(train_set).long()
            valid_set = torch.tensor(valid_set).long()

            loss_f = F.binary_cross_entropy

            for epoch in range(args.external_epochs):

                net, classifier_model, fold_best_metrics = train(
                    args, net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg,
                    train_set, valid_set, metabolite_count, reaction_count, loss_f,
                    training_data=(
                        train_edges_pos_set, train_weight_pos_set, train_edges_neg_set, train_weight_neg_set),
                    validation_data=(
                        valid_edges_pos_set, valid_weight_pos_set, valid_edges_neg_set, valid_weight_neg_set),
                    optimizer=[optimizer],
                    epochs=args.internal_epochs,
                    batch_size=current_batch_size,
                    fold=fold,
                    model_name=model_name)

                for metric_name in metric_collector.keys():
                    metric_collector[metric_name].append(fold_best_metrics[metric_name])

                all_folds_best_metrics.append(fold_best_metrics)

                print(f"\n[Cross-Validation Results - Fold {fold}]")
                print(f"  Best Epoch: {fold_best_metrics['epoch']}")
                print(f"  AUROC:     {fold_best_metrics['valid_auroc']:.4f}")
                print(f"  F1-Score:  {100 * fold_best_metrics['valid_f1']:.2f}%")


            fold_assignments_file = os.path.join(args.save_path, 'fold_assignments.csv')
            with open(fold_assignments_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['bigg_id', 'fold'])
                writer.writeheader()
                for assignment in fold_assignments:
                    writer.writerow(assignment)


        try:
            df_metrics = pd.DataFrame(all_folds_best_metrics)
            df_metrics.insert(0, 'Fold', [f'Fold {i}' for i in range(len(all_folds_best_metrics))])
            columns_map = {
                'valid_acc': 'Accuracy',
                'valid_precision': 'Precision',
                'valid_recall': 'Recall',
                'valid_f1': 'F1-Score',
                'valid_mcc': 'MCC',
                'valid_auroc': 'AUROC',
                'valid_auprc': 'AUPRC',
            }
            df_metrics = df_metrics.rename(columns=columns_map)
            desired_order = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUROC', 'AUPRC']
            final_columns = [col for col in desired_order if col in df_metrics.columns]
            df_metrics = df_metrics[final_columns]
            mean_row = df_metrics.iloc[:, 1:].mean(numeric_only=True)
            std_row = df_metrics.iloc[:, 1:].std(numeric_only=True)
            mean_row['Fold'] = 'Average'
            std_row['Fold'] = 'Std Dev'
            df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)
            cols = ['Fold'] + [c for c in df_metrics.columns if c != 'Fold']
            df_metrics = df_metrics[cols]
            excel_filename = f'{model_name}.xlsx'
            excel_save_path = os.path.join(args.save_path, excel_filename)

            try:
                df_metrics.to_excel(excel_save_path, index=False)
                print(f"\n[Success] Metrics Excel saved to: {excel_save_path}")
            except Exception as e:
                if "No module named 'openpyxl'" in str(e) or "ImportError" in str(e):
                    csv_filename = f'{model_name}.csv'
                    csv_save_path = os.path.join(args.save_path, csv_filename)
                    df_metrics.to_csv(csv_save_path, index=False)
                    print(f"\n[Warning] Saved as CSV instead: {csv_save_path}")
                else:
                    raise e

        except Exception as e:
            print(f"\n[Error] Failed to save metrics file: {e}")

        print(f"\n{'=' * 20} Final Average Results ({args.k_fold}-Fold) {'=' * 20}")
        if len(metric_collector['valid_acc']) > 0:
            print(
                f"  Accuracy:  {100 * np.mean(metric_collector['valid_acc']):.2f}% ± {100 * np.std(metric_collector['valid_acc']):.2f}%")
            print(
                f"  Precision:  {100 * np.mean(metric_collector['valid_precision']):.2f}% ± {100 * np.std(metric_collector['valid_precision']):.2f}%")
            print(
                f"  Recall:  {100 * np.mean(metric_collector['valid_recall']):.2f}% ± {100 * np.std(metric_collector['valid_recall']):.2f}%")
            print(
                f"  F1-Score:  {100 * np.mean(metric_collector['valid_f1']):.2f}% ± {100 * np.std(metric_collector['valid_f1']):.2f}%")
            print(
                f"  MCC:  {100 * np.mean(metric_collector['valid_mcc']):.2f}% ± {100 * np.std(metric_collector['valid_mcc']):.2f}%")
            print(
                f"  AUROC:     {np.mean(metric_collector['valid_auroc']):.4f} ± {np.std(metric_collector['valid_auroc']):.4f}")
            print(
                f"  AUPRC:     {np.mean(metric_collector['valid_auprc']):.4f} ± {np.std(metric_collector['valid_auprc']):.4f}")
        print("=" * 60)