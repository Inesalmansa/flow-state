import math
import torch
from torch import nn

# E(n)-Equivariant Graph Neural Networks in torus space
# https://github.com/lucidrains/egnn-pytorch/tree/main

class TorusEGNN (nn.Module):
    def __init__(self, feat_dim: int, edge_dim:int = 0, hidden_features:int = 128, dropout:float = 0.1):
        super().__init__()
        # feat_dim:

        edge_input_dim = feat_dim * 2 + 1 + edge_dim        

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_features // 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_features // 2, hidden_features),
            nn.Dropout(dropout),
            nn.SiLU(),
        )

        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

        # self.node_norm = nn.LayerNorm(feat_dim) if norm_feats else nn.Identity()
        # self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features// 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_features // 2, feat_dim),
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_features // 2, 1),
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, feats, coors, edges = None, mask = None):
        b, n, d, device = *feats.shape, feats.device

        if mask is not None:
            num_nodes = mask.sum(dim = -1)

        # rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_coors = coors[:,:,None,:] - coors[:,None,:,:]                                   # b i 1 d - b 1 j d = b i j d
        rel_coors = rel_coors - 2 * math.pi * torch.round(rel_coors / (2 * math.pi))        # wrap at boundary
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        i = j = n

        feats_j, feats_i = feats[:,None,:,:], feats[:,:,None,:]
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)

        if edges is not None:
            edge_input = torch.cat((edge_input, edges), dim = -1)

        # message of edges
        m_ij = self.edge_mlp(edge_input)
        m_ij = m_ij * self.edge_gate(m_ij)                          # b i j h

        # message of coordinates
        coor_weights = self.coors_mlp(m_ij)
        coor_weights = torch.squeeze(coor_weights, dim=-1)          # b i j

        # rel_coors = self.coors_norm(rel_coors)

        if mask is not None:
            coor_weights.masked_fill_(~mask.bool(), 0.)

        # aggregation and update of coordinates
        coors_out = torch.einsum('b i j, b i j c -> b i c', coor_weights, rel_coors / (rel_dist + 2 * math.pi)) + coors

        if mask is not None:
            # m_ij_mask = rearrange(mask, '... -> ... ()')
            m_ij_mask = mask[..., None]
            m_ij = m_ij.masked_fill(~m_ij_mask.bool(), 0.)

        # aggregation of edges
        m_i = m_ij.sum(dim = -2)
        
        # update of nodes
        # normed_feats = self.node_norm(feats)
        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors_out
    

class FullEquivariantGraphNetwork (nn.Module):
    def __init__(self, num_node: int, out_dim: int,num_layers: int, edge_dim: int=0, feat_dim: int=2, hidden_dim: int=128, dropout: float=0.1, preprocessing=None):
        super().__init__()

        # mask out self connection
        self.register_buffer('mask', (1 - torch.eye(num_node))[None, ...])
        self.layers = nn.Sequential(
            *[TorusEGNN(feat_dim=feat_dim, edge_dim=edge_dim, hidden_features=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        # self.output_fc = nn.Linear(feat_dim+coor_dim, coor_dim)
        #
        # nn.init.xavier_normal_(self.output_fc.weight)
        # nn.init.zeros_(self.output_fc.weight)
        self.hidden_features = hidden_dim
        self.preprocessing = preprocessing
        self.pool = nn.AdaptiveAvgPool1d(1)  # Aggregates node features
        self.final_linear = nn.Linear(feat_dim, out_dim)  # Maps node features to spline parameters
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, coors, edges = None):
        coors = coors.unsqueeze(-1)
        if self.preprocessing is None:
            feats = coors
        else:
            feats = self.preprocessing(coors)
        b, n, device = feats.shape[0], feats.shape[1], feats.device

        # coors_residual = coors
        # go through layers
        # coor_changes = [coors]

        for egnn in self.layers:
            feats, coors = egnn(feats, coors, edges = edges, mask = self.mask)
            # coor_changes.append(coors)

        feats_perm = feats.transpose(1, 2)  # (B, feat_dim, num_nodes)
        pooled = self.pool(feats_perm).squeeze(-1)  # (B, feat_dim)
        out = self.final_linear(pooled)  # (B, 194)
        out = self.norm(out)

        # if return_coor_changes:

        #     return feats, coors, coor_changes
        
        # coors = self.output_fc(torch.cat((feats, coors), dim=-1))
        # coors -= coors_residual
        return out

# batch_size = 128       # 批大小
# num_nodes = 3          # 节点数
# coor_dim = 1           # 坐标维度
# feat_dim = 4           # 节点特征维度
# hidden_dim = 64        # 每个 GNN block 的隐藏层维度
# num_layers = 2         # GNN block 数量
# out_dim = 194          # 最终输出维度
#
# # 构造输入数据
# # 坐标假设在 [0, 2π) 内
# coors = torch.rand(batch_size, num_nodes, coor_dim) * (2 * math.pi)
# # 特征随机生成
# feats = torch.rand(batch_size, num_nodes, feat_dim)
#
# # 实例化模型
# model = FullEquivariantGraphNetwork(
#     num_node=num_nodes,
#     feat_dim=feat_dim,
#     out_dim=out_dim,
#     edge_dim=0,
#     hidden_dim=hidden_dim,
#     num_layers=num_layers,
#     dropout=0.1
# )

# # 前向传播，得到输出 (形状应为 (128, 194))
# output = model(feats,coors)
# print("Output shape:", output.shape)
