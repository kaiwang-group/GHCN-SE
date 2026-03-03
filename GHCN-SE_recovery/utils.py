import dhg
import xml.etree.ElementTree as ET
import csv
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef


def process_data():
    # 1. 读取基础文件
    metabolite = pd.read_csv('metabolite.csv')
    reaction = pd.read_csv('reaction.csv')
    reaction_count = reaction['bigg_id'].count()
    metabolite_count = len(metabolite)

    # 2. 创建超图 (调库)
    hg_pos = create_hypergraph(pd.read_csv('incidence_matrix_pos.csv'), metabolite_count)
    hg_neg = create_hypergraph(pd.read_csv('incidence_matrix_neg.csv'), metabolite_count)

    # 3. 构建 PyTorch 原生稀疏邻接矩阵 (SparseGCN用)
    if os.path.exists('mmi.csv'):
        mmi = pd.read_csv('mmi.csv')
        src_ids = mmi['substrate_id'].values
        dst_ids = mmi['product_id'].values
    else:
        src_ids = np.array([])
        dst_ids = np.array([])

    all_nodes = np.arange(metabolite_count)
    src_ids = np.concatenate([src_ids, all_nodes])
    dst_ids = np.concatenate([dst_ids, all_nodes])

    indices = torch.tensor([dst_ids, src_ids], dtype=torch.long)
    degrees = torch.bincount(indices[0], minlength=metabolite_count).float()
    degrees[degrees == 0] = 1.0
    norm = 1.0 / degrees
    values = norm[indices[0]]

    adj_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        (metabolite_count, metabolite_count),
        dtype=torch.float32
    )

    print(f"使用完全可学习的随机嵌入: 生成 {metabolite_count} 个索引...")
    metabolite_features = torch.arange(metabolite_count, dtype=torch.long)

    return adj_matrix, hg_pos, hg_neg, metabolite_features, reaction_count, metabolite_count


def create_hypergraph(file, count):
    # 构建 DHG Hypergraph 对象
    lists = []
    for _, group in file.groupby(file.columns[0]):
        hyperedge = list(group[group.columns[1]])
        lists.append(hyperedge)
    hg = dhg.Hypergraph(count, lists)
    return hg


def read_xml(xml_content):
    root = ET.fromstring(xml_content)
    ns = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
        'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2'
    }
    list_of_reactions = root.find('.//sbml:model/sbml:listOfReactions', ns)
    reactions = []
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        bigg_id = reaction.get('id').replace('R_', '')
        reactions.append(bigg_id)

    with open('reaction.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'hyperedge_id'])
        for index, bigg_id in enumerate(reactions):
            writer.writerow([bigg_id, index])

    metabolites = set()
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        for reactant in reaction.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            species_id = reactant.get('species').replace('M_', '')
            metabolites.add(species_id)
        for product in reaction.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            species_id = product.get('species').replace('M_', '')
            metabolites.add(species_id)

    metabolites = sorted(list(metabolites))
    with open('metabolite.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bigg_id', 'node_id'])
        for index, bigg_id in enumerate(metabolites):
            writer.writerow([bigg_id, index])

    with open('data/universal_model.json', 'r') as file:
        model_data = json.load(file)
    metabolites_df = pd.read_csv('metabolite.csv')
    metabolite_list = metabolites_df['bigg_id'].unique()
    metabolite_relationships = []
    for reaction in model_data['reactions']:
        metabolites_in_reaction = reaction['metabolites']
        for metabolite in metabolite_list:
            if metabolite in metabolites_in_reaction:
                for product, stoichiometry in metabolites_in_reaction.items():
                    if stoichiometry > 0:
                        metabolite_relationships.append([metabolite, product])

    relationship_df = pd.DataFrame(metabolite_relationships, columns=['Metabolite', 'Product'])
    relationship_df = relationship_df.drop_duplicates()
    relationship_df.to_csv('metabolite_relationships.csv', index=False)
    metabolites_df = pd.read_csv('metabolite.csv')
    relationships_df = pd.read_csv('metabolite_relationships.csv')
    metabolite_to_id = pd.Series(metabolites_df.node_id.values, index=metabolites_df.bigg_id).to_dict()
    updated_relationships = []
    for _, row in relationships_df.iterrows():
        if row['Metabolite'] != row['Product']:
            substrate_id = metabolite_to_id.get(row['Metabolite'])
            product_id = metabolite_to_id.get(row['Product'])
            if substrate_id is not None and product_id is not None:
                updated_relationships.append({
                    'Metabolite': row['Metabolite'],
                    'Product': row['Product'],
                    'substrate_id': substrate_id,
                    'product_id': product_id
                })

    updated_relationships_df = pd.DataFrame(updated_relationships)
    updated_relationships_df.to_csv('mmi.csv', index=False)
    reaction_df = pd.read_csv('reaction.csv')
    metabolite_df = pd.read_csv('metabolite.csv')
    reaction_to_id = pd.Series(reaction_df.hyperedge_id.values, index=reaction_df.bigg_id).to_dict()
    metabolite_to_id = pd.Series(metabolite_df.node_id.values, index=metabolite_df.bigg_id).to_dict()
    incidence_matrix_data = []

    def parse_xml(file_path):
        root = ET.fromstring(file_path)
        ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}
        for reaction in root.findall('.//sbml:listOfReactions/sbml:reaction', ns):
            reaction_id = reaction.attrib['id'].replace('R_', '')
            hyperedge_id = reaction_to_id.get(reaction_id)
            if hyperedge_id is not None:
                reactants = reaction.find('sbml:listOfReactants', ns)
                products = reaction.find('sbml:listOfProducts', ns)
                if reactants is not None:
                    for reactant in reactants.findall('sbml:speciesReference', ns):
                        metabolite_id = reactant.attrib['species'].replace('M_', '')
                        node_id = metabolite_to_id.get(metabolite_id)
                        if node_id is not None:
                            incidence_matrix_data.append({
                                'hyperedge_id': hyperedge_id,
                                'node_id': node_id
                            })
                if products is not None:
                    for product in products.findall('sbml:speciesReference', ns):
                        metabolite_id = product.attrib['species'].replace('M_', '')
                        node_id = metabolite_to_id.get(metabolite_id)
                        if node_id is not None:
                            incidence_matrix_data.append({
                                'hyperedge_id': hyperedge_id,
                                'node_id': node_id
                            })

    parse_xml(xml_content)

    incidence_matrix_df = pd.DataFrame(incidence_matrix_data)
    incidence_matrix_df.to_csv('incidence_matrix_pos.csv', index=False)
    df = pd.read_csv('incidence_matrix_pos.csv')
    np.random.seed(42)
    neg_data = []
    hyperedge_ids = df['hyperedge_id'].unique()

    total_metabolites = len(metabolites)
    for hyperedge_id in hyperedge_ids:
        metabolites = df[df['hyperedge_id'] == hyperedge_id]['node_id'].tolist()
        all_possible_metabolites = set(range(0, total_metabolites))
        existing_metabolites = set(metabolites)
        possible_replacements = list(all_possible_metabolites - existing_metabolites)
        random_replacements = np.random.choice(possible_replacements, len(metabolites), replace=False)
        for metabolite in random_replacements:
            neg_data.append({'hyperedge_id': hyperedge_id, 'node_id': metabolite})

    neg_df = pd.DataFrame(neg_data)
    neg_df.to_csv('incidence_matrix_neg.csv', index=False)
    return


def add_padding_idx(vec):
    if len(vec.shape) == 1:
        return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
                           for v in tqdm(vec)])
    else:
        vec = np.asarray(vec) + 1
        vec = np.sort(vec, axis=-1)
        return vec.astype('int')


def np2tensor_hyper(vec, dtype):
    vec = np.asarray(vec)
    if len(vec.shape) == 1:
        return [torch.as_tensor(v, dtype=dtype) for v in vec]
    else:
        return torch.as_tensor(vec, dtype=dtype)


def roc_auc_cuda(y_true, y_pred):
    try:
        y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
        y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
        return roc_auc_score(
            y_true, y_pred), average_precision_score(
            y_true, y_pred)
    except BaseException:
        return 0.0, 0.0


def accuracy(output, target):
    pred = output >= 0.5
    truth = target >= 0.5
    acc = torch.sum(pred.eq(truth))
    acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
    return acc


def pass_(x):
    return x