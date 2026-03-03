import torch
import os
import glob
import argparse
import numpy as np
import pandas as pd
import math
import warnings
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score)
from dhg.random import set_seed
from model import Model, Classifier
from utils import read_xml, process_data

warnings.filterwarnings("ignore")

DATASET_NAME = 'iAF1260'
PROJECT_ROOT = '/data/stu1/QJJpycharm_project/SGHG-SE_prediction'

ARGS_DEFAULTS = {
    'in_dim': 512,
    'h_dim': 256,
    'out_dim': 64,
    'num_dgnn': 2,
    'num_hgnn': 1,
    'dimensions': 64,
    'bottle_neck': 64,
    'diag': 'True',
    'cuda': 0,
    'batch_size': 96
}


def get_device(gpu_index):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def test_fold(net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg,
              test_indices, edges_pos, edges_neg, weights_pos, weights_neg,
              batch_size, device):
    net.eval()
    classifier_model.eval()

    test_edges_pos_set = [edges_pos[i] for i in test_indices]
    test_edges_neg_set = [edges_neg[i] for i in test_indices]

    test_weight_pos_set = weights_pos[test_indices]
    test_weight_neg_set = weights_neg[test_indices]

    max_len = max(len(edge) for edge in test_edges_pos_set)
    if max_len == 0: max_len = 1

    test_edges_pos_padded = [edge + [0] * (max_len - len(edge)) for edge in test_edges_pos_set]
    test_edges_neg_padded = [edge + [0] * (max_len - len(edge)) for edge in test_edges_neg_set]

    test_edges_pos_tensor = torch.tensor(test_edges_pos_padded, dtype=torch.long, device=device)
    test_edges_neg_tensor = torch.tensor(test_edges_neg_padded, dtype=torch.long, device=device)
    test_weight_pos_tensor = torch.tensor(test_weight_pos_set, device=device)
    test_weight_neg_tensor = torch.tensor(test_weight_neg_set, device=device)

    pred_list, label_list, score_list = [], [], []

    with torch.no_grad():
        X1, X2, Y_pos, Y_neg = net(m_emb, adj_matrix, hg_pos, hg_neg)

        if X1.dim() > 2: X1 = X1.squeeze()
        classifier_model.set_node_embedding(X1)

        num_batches = int(math.floor(len(test_edges_pos_tensor) / batch_size))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            batch_x_pos = test_edges_pos_tensor[start_idx: end_idx]
            batch_x_neg = test_edges_neg_tensor[start_idx: end_idx]

            batch_x = torch.cat([batch_x_pos, batch_x_neg])
            batch_y = torch.cat([
                torch.ones(len(batch_x_pos), device=device),
                torch.zeros(len(batch_x_neg), device=device)
            ])

            pred_batch = classifier_model(batch_x, return_recon=False)
            pred_batch = pred_batch.squeeze(1)

            pred_list.append((pred_batch >= 0.5).float())
            label_list.append(batch_y)
            score_list.append(pred_batch)

    if len(pred_list) == 0:
        return {}

    pred = torch.cat(pred_list)
    label = torch.cat(label_list)
    scores = torch.cat(score_list)

    y_np = label.cpu().numpy()
    pred_np = pred.cpu().numpy()
    scores_np = scores.cpu().numpy()

    metrics = {
        'Accuracy': accuracy_score(y_np, pred_np),
        'Precision': precision_score(y_np, pred_np, zero_division=0),
        'Recall': recall_score(y_np, pred_np, zero_division=0),
        'F1-Score': f1_score(y_np, pred_np, zero_division=0),
        'MCC': matthews_corrcoef(y_np, pred_np),
        'AUROC': roc_auc_score(y_np, scores_np),
        'AUPRC': average_precision_score(y_np, scores_np)
    }
    return metrics


if __name__ == "__main__":
    set_seed(0)

    args = argparse.Namespace(**ARGS_DEFAULTS)
    device = get_device(args.cuda)

    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints/model', DATASET_NAME)
    fold_file_path = os.path.join(checkpoint_dir, 'fold_assignments.csv')

    xml_pattern = os.path.join('BiGG Models', f'*{DATASET_NAME}*.xml')
    xml_files = glob.glob(xml_pattern)

    if not xml_files:
        print(f"[Error] No XML file found for {DATASET_NAME} in 'BiGG Models/'")
        exit(1)

    xml_file = xml_files[0]
    print(f"{'=' * 60}")
    print(f"Testing Model: {DATASET_NAME}")
    print(f"XML File:      {xml_file}")
    print(f"Checkpoints:   {checkpoint_dir}")
    print(f"Fold File:     {fold_file_path}")
    print(f"{'=' * 60}\n")

    print(">>> Processing Data (Reconstructing Hypergraphs)...")

    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    read_xml(xml_content)

    df_pos = pd.read_csv('incidence_matrix_pos.csv')
    df_neg = pd.read_csv('incidence_matrix_neg.csv')
    reaction = pd.read_csv('reaction.csv')

    reaction_count_total = len(reaction)

    pos_dict = df_pos.groupby('hyperedge_id')['node_id'].apply(list).to_dict()
    neg_dict = df_neg.groupby('hyperedge_id')['node_id'].apply(list).to_dict()

    edges_pos = [pos_dict.get(i, []) for i in range(reaction_count_total)]
    edges_neg = [neg_dict.get(i, []) for i in range(reaction_count_total)]

    edges_weight_pos = np.ones(len(edges_pos), dtype='float32')
    edges_weight_neg = np.ones(len(edges_neg), dtype='float32')

    adj_matrix, hg_pos, hg_neg, initial_features, reaction_count, metabolite_count = process_data()

    m_emb = initial_features.to(device)
    adj_matrix = adj_matrix.to(device)
    hg_pos = hg_pos.to(device)
    hg_neg = hg_neg.to(device)

    if not os.path.exists(fold_file_path):
        print(f"[Error] Fold assignments file not found: {fold_file_path}")
        exit(1)

    print(f">>> Loading fold assignments from {fold_file_path}")
    fold_df = pd.read_csv(fold_file_path)

    bigg_id_to_idx = dict(zip(reaction['bigg_id'], reaction.index))

    folds_indices = {i: [] for i in range(5)}
    for _, row in fold_df.iterrows():
        b_id = row['bigg_id']
        f_id = int(row['fold'])
        if b_id in bigg_id_to_idx:
            idx = bigg_id_to_idx[b_id]
            folds_indices[f_id].append(idx)
        else:
            print(f"Warning: Reaction {b_id} in fold file not found in current reaction list.")

    all_metrics = []

    print("\nStarting Inference Loop...\n")

    for fold in range(5):
        model_path = os.path.join(checkpoint_dir, f'trained_model_fold{fold}_{DATASET_NAME}.pth')

        if not os.path.exists(model_path):
            print(f"[Warning] Model file for Fold {fold} not found. Skipping.")
            continue

        net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_dgnn, args.num_hgnn,
                    num_nodes=metabolite_count, use_bn=True).to(device)

        classifier_model = Classifier(
            n_head=8, d_model=args.dimensions, d_k=16, d_v=16, node_embedding=None,
            metabolite_count=metabolite_count, diag_mask=args.diag, bottle_neck=args.bottle_neck,
            device=device
        ).to(device)

        print(f"Testing Fold {fold} | Model: {os.path.basename(model_path)}")
        checkpoint = torch.load(model_path, map_location=device)

        net_state = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        cls_state = {k.replace('module.', ''): v for k, v in checkpoint['classifier_model'].items()}

        net.load_state_dict(net_state)
        classifier_model.load_state_dict(cls_state)

        test_indices = folds_indices[fold]

        if len(test_indices) == 0:
            print(f"  No test data found for Fold {fold}.")
            continue

        metrics = test_fold(
            net, classifier_model, m_emb, adj_matrix, hg_pos, hg_neg,
            test_indices, edges_pos, edges_neg, edges_weight_pos, edges_weight_neg,
            args.batch_size, device
        )

        if not metrics:
            print(f"  No valid batches processed for Fold {fold} (Check batch size vs data size).")
            continue

        metrics['Fold'] = fold
        all_metrics.append(metrics)

        print(f"  -> AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1-Score']:.4f} | Acc: {metrics['Accuracy']:.4f}")
        print("-" * 40)

    if all_metrics:
        df_res = pd.DataFrame(all_metrics)
        cols = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUROC', 'AUPRC']
        df_res = df_res[cols]

        print("\n" + "=" * 25 + " FINAL RESULTS " + "=" * 25)
        print(df_res.to_string(index=False))
        print("-" * 65)

        mean_val = df_res.iloc[:, 1:].mean()
        std_val = df_res.iloc[:, 1:].std()

        print(f"\nAverage Results ({len(df_res)}-Fold):")
        for metric in mean_val.index:
            print(f"  {metric:<10}: {mean_val[metric]:.4f} ± {std_val[metric]:.4f}")
        print("=" * 65)
    else:
        print("\nNo metrics collected. Check paths and data files.")