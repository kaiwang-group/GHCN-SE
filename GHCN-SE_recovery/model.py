from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
import dhg.nn as dhgnn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 模块 1: 基础组件 (保持不变)
# =========================================================================

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

        # 内部聚合器
        self.aggregator = dhgnn.HGNNPConv(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bn=use_bn,
            drop_rate=drop_rate
        )

    def forward(self, X, hg):
        # 1. 聚合计算
        X_out = self.aggregator(X, hg)

        # 2. 接口适配 (返回 Node特征 和 占位符)
        return X_out, None


class SEContextGating(nn.Module):
    """
    【方案 C 核心】SE-Block / Context Gating
    """

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


# =========================================================================
# 模块 2: 主模型架构 (【修改核心】：改为并联架构)
# =========================================================================

class Model(nn.Module):
    def __init__(self, in_channels: int,  # GCN输入: 512
                 hid_channels: int,  # GCN输出: 256
                 out_channels: int,  # HGNN输出: 64
                 num_dgnn: int,
                 num_hgnn: int,
                 num_nodes: int,
                 use_bn: bool = False) -> None:
        super().__init__()

        # 1. 基础 Embedding (固定 1024)
        self.embedding = nn.Embedding(num_nodes, 1024)

        self.num_dgnn = num_dgnn
        self.num_hgnn = num_hgnn

        # ============================================================
        # 分支 A: SparseGCN 组件
        # ============================================================
        # 适配器 A: 1024 -> 512 (in_channels)
        self.adapter_gcn = nn.Linear(1024, in_channels)

        self.gcn_layers = nn.ModuleList()
        # GCN 第1层: 512 -> 256
        self.gcn_layers.append(
            SparseGCNLayer(in_channels, hid_channels, activation=F.relu))

        # GCN 后续层: 256 -> 256
        for i in range(self.num_dgnn - 1):
            self.gcn_layers.append(
                SparseGCNLayer(hid_channels, hid_channels, activation=F.relu))

        # GCN 分支最终输出: 256 (hid_channels)

        # ============================================================
        # 分支 B: HGNN 组件 (已修改)
        # ============================================================
        # 适配器 B: 1024 -> 256 (此处修改：原为映射到64，现改为256)
        # 我们使用 hid_channels (256) 作为 HGNN 的输入维度
        self.hgnn_input_dim = 256
        self.adapter_hgnn = nn.Linear(1024, self.hgnn_input_dim)

        self.hgnn_layers = nn.ModuleList()

        # HGNN 层构建逻辑：
        # 输入是 256，要求最终输出是 64 (out_channels)
        # 策略：第一层直接降维 256 -> 64，后续保持 64 -> 64
        # 这样既满足了输入要求，也满足了输出要求

        if self.num_hgnn > 0:
            # 第一层: 256 -> 64
            self.hgnn_layers.append(
                LightweightAllSetLayer(self.hgnn_input_dim, out_channels, use_bn=use_bn))

            # 后续层: 64 -> 64
            for i in range(self.num_hgnn - 1):
                self.hgnn_layers.append(
                    LightweightAllSetLayer(out_channels, out_channels, use_bn=use_bn))

        # HGNN 分支最终输出: 64 (out_channels)

        # ============================================================
        # 融合层 (Fusion)
        # ============================================================
        # 输入: GCN输出(256) + HGNN输出(64) = 320
        # 输出: 64 (out_channels)
        self.fusion_dim = hid_channels + out_channels
        self.fusion_layer = nn.Linear(self.fusion_dim, out_channels)

    def forward(self, m_emb: torch.Tensor, adj, hg_pos: dhg.Hypergraph, hg_neg: dhg.Hypergraph) -> Any:
        # 0. 原始 Embedding
        x_raw = self.embedding(m_emb)  # [N, 1024]

        # ==================== 分支 A: GCN ====================
        # 1. 映射到 512
        x_gcn = F.relu(self.adapter_gcn(x_raw))  # [N, 512]

        # 2. GCN 卷积 (最终输出 256)
        for layer in self.gcn_layers:
            x_gcn = layer(adj, x_gcn)
            # x_gcn shape: [N, 256]

        # ==================== 分支 B: HGNN ====================
        # 1. 映射到 256 (已修改)
        x_hgnn_base = F.relu(self.adapter_hgnn(x_raw))  # [N, 256]

        # 2. HGNN 卷积 (分别处理 Pos 和 Neg)
        # 输入 256 -> 第一层 -> 64 -> 后续层 -> 64
        x_hgnn_pos = x_hgnn_base
        x_hgnn_neg = x_hgnn_base

        # Pos Path
        for layer in self.hgnn_layers:
            x_hgnn_pos, _ = layer(x_hgnn_pos, hg_pos)

        # Neg Path
        for layer in self.hgnn_layers:
            x_hgnn_neg, _ = layer(x_hgnn_neg, hg_neg)
        # x_hgnn shape: [N, 64]

        # ==================== 融合 ====================
        # 拼接: [N, 256] cat [N, 64] -> [N, 320]
        x_concat_pos = torch.cat([x_gcn, x_hgnn_pos], dim=1)
        x_concat_neg = torch.cat([x_gcn, x_hgnn_neg], dim=1)

        # 映射: [N, 320] -> [N, 64]
        out_pos = self.fusion_layer(x_concat_pos)
        out_neg = self.fusion_layer(x_concat_neg)

        return out_pos, out_neg, None, None


# =========================================================================
# 模块 3: 分类器 (方案 C 纯享版) (保持不变)
# =========================================================================

class Classifier(nn.Module):

    def __init__(
            self,
            n_head, d_model, d_k, d_v, node_embedding, metabolite_count, diag_mask, bottle_neck, **args):
        super().__init__()

        self.device = args.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 1. Embedding
        self.node_embedding = node_embedding
        self.is_tensor_embedding = isinstance(node_embedding, torch.Tensor)
        if self.is_tensor_embedding:
            self.node_embedding = self.node_embedding.to(args.get('device', 'cuda'))
        elif node_embedding is None:
            n_nodes = metabolite_count
            self.node_embedding = torch.randn(n_nodes, bottle_neck).to(args.get('device', 'cuda'))
            self.is_tensor_embedding = True

        # 2. SE-Gating (特征去噪)
        self.se_gating = SEContextGating(channel=d_model, reduction=4)

        # 3. LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)

        # 4. 【关键修正】MLP 分类头
        # 替代了方案 E 的 temperature 参数，回归经典的 MLP 预测
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
        """
        x: [Batch, Seq_Len]
        """
        x = x.long()
        non_pad_mask = get_non_pad_mask(x)  # [Batch, Seq, 1]

        # 1. 获取特征
        dynamic = self.get_node_embeddings(x)  # [Batch, Seq, Dim]

        # 2. SE-Gating 去噪
        refined = self.se_gating(dynamic)  # [Batch, Seq, Dim]
        refined = self.layer_norm(refined)

        # 3. 聚合 (Sum Pooling)
        # 将 Mask 为 0 的部分（padding）剔除
        refined_masked = refined * non_pad_mask

        # 将整个反应的特征加起来，得到反应向量
        reaction_vector = refined_masked.sum(dim=1)  # [Batch, Dim]

        # 4. MLP 预测
        # 【关键修正】这里不再使用 temperature，而是调用 predict_mlp
        logits = self.predict_mlp(reaction_vector)  # [Batch, 1]
        output = torch.sigmoid(logits)  # [Batch, 1]

        return output


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)