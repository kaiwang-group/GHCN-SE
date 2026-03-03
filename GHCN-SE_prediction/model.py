from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
import dhg.nn as allset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SparseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, adj, x):
        h = torch.sparse.mm(adj, x)
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        return h


class LightweightAllSetLayer(nn.Module):


    def __init__(self, in_channels, out_channels, use_bn=False, drop_rate=0.5):
        super(LightweightAllSetLayer, self).__init__()


        self.aggregator = allset.HGNNPConv(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bn=use_bn,
            drop_rate=drop_rate
        )

    def forward(self, X, hg):

        X_out = self.aggregator(X, hg)


        return X_out, None


class SEContextGating(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEContextGating, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # Excitation
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        b, s, c = x.size()
        y = x.permute(0, 2, 1)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y.expand_as(x)  # Scale


class Model(nn.Module):
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 num_gcn: int,
                 num_hg: int,
                 num_nodes: int,
                 use_bn: bool = False) -> None:
        super().__init__()


        self.embedding = nn.Embedding(num_nodes, 1024)

        self.num_gcn = num_gcn
        self.num_hg = num_hg


        self.adapter_gcn = nn.Linear(1024, in_channels)

        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(
            SparseGCNLayer(in_channels, hid_channels, activation=F.relu))

        for i in range(self.num_gcn - 1):
            self.gcn_layers.append(
                SparseGCNLayer(hid_channels, hid_channels, activation=F.relu))



        self.hg_input_dim = 256
        self.adapter_hg = nn.Linear(1024, self.hg_input_dim)

        self.hg_layers = nn.ModuleList()



        if self.num_hg > 0:

            self.hg_layers.append(
                LightweightAllSetLayer(self.hg_input_dim, out_channels, use_bn=use_bn))


            for i in range(self.num_hg - 1):
                self.hg_layers.append(
                    LightweightAllSetLayer(out_channels, out_channels, use_bn=use_bn))



        self.fusion_dim = hid_channels + out_channels
        self.fusion_layer = nn.Linear(self.fusion_dim, out_channels)

    def forward(self, m_emb: torch.Tensor, adj, hg_pos: dhg.Hypergraph, hg_neg: dhg.Hypergraph) -> Any:

        x_raw = self.embedding(m_emb)  # [N, 1024]


        x_gcn = F.relu(self.adapter_gcn(x_raw))  # [N, 512]


        for layer in self.gcn_layers:
            x_gcn = layer(adj, x_gcn)



        x_hg_base = F.relu(self.adapter_hg(x_raw))  # [N, 256]


        x_hg_pos = x_hg_base
        x_hg_neg = x_hg_base


        for layer in self.hg_layers:
            x_hg_pos, _ = layer(x_hg_pos, hg_pos)


        for layer in self.hg_layers:
            x_hg_neg, _ = layer(x_hg_neg, hg_neg)


        x_concat_pos = torch.cat([x_gcn, x_hg_pos], dim=1)
        x_concat_neg = torch.cat([x_gcn, x_hg_neg], dim=1)


        out_pos = self.fusion_layer(x_concat_pos)
        out_neg = self.fusion_layer(x_concat_neg)

        return out_pos, out_neg, None, None


class Classifier(nn.Module):

    def __init__(
            self,
            n_head, d_model, d_k, d_v, node_embedding, metabolite_count, diag_mask, bottle_neck, **args):
        super().__init__()

        self.device = args.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


        self.node_embedding = node_embedding
        self.is_tensor_embedding = isinstance(node_embedding, torch.Tensor)
        if self.is_tensor_embedding:
            self.node_embedding = self.node_embedding.to(args.get('device', 'cuda'))
        elif node_embedding is None:
            n_nodes = metabolite_count
            self.node_embedding = torch.randn(n_nodes, bottle_neck).to(args.get('device', 'cuda'))
            self.is_tensor_embedding = True


        self.se_gating = SEContextGating(channel=d_model, reduction=4)


        self.layer_norm = nn.LayerNorm(d_model)

        self.predict_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )

    def set_node_embedding(self, new_embedding):
        self.node_embedding = new_embedding
        self.is_tensor_embedding = isinstance(new_embedding, torch.Tensor)

    def get_node_embeddings(self, x):
        x_flat = x.view(-1)
        if self.is_tensor_embedding:
            embedded_x = self.node_embedding[x_flat]
        else:
            embedded_x, _ = self.node_embedding(x_flat)
        return embedded_x.view(x.shape[0], x.shape[1], -1)

    def forward(self, x, mask=None, get_outlier=None, return_recon=False):

        x = x.long()
        non_pad_mask = get_non_pad_mask(x)  # [Batch, Seq, 1]


        dynamic = self.get_node_embeddings(x)  # [Batch, Seq, Dim]


        refined = self.se_gating(dynamic)  # [Batch, Seq, Dim]
        refined = self.layer_norm(refined)


        refined_masked = refined * non_pad_mask


        reaction_vector = refined_masked.sum(dim=1)  # [Batch, Dim]


        logits = self.predict_mlp(reaction_vector)  # [Batch, 1]
        output = torch.sigmoid(logits)  # [Batch, 1]

        return output


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)