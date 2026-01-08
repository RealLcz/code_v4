import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch.autograd import Variable
from torch_scatter import scatter



def make_one_hot(labels, C):
    labels = labels.long()
    one_hot = F.one_hot(labels, num_classes=C).float()
    return one_hot

class GATConvE(MessagePassing):
    def __init__(self, args, emb_dim, n_ntype, n_etype, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype
        # self.edge_encoder = edge_encoder
        self.gnn_edge_dim = args.gnn_edge_dim
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(265, self.gnn_edge_dim), torch.nn.LayerNorm(self.gnn_edge_dim), torch.nn.ReLU(), torch.nn.Linear(self.gnn_edge_dim, self.gnn_edge_dim))

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        # self.linear_key = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        # self.linear_msg = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        self.linear_key = nn.Linear(emb_dim + self.gnn_edge_dim, head_count * self.dim_per_head) # [4096+1024,
        self.linear_msg = nn.Linear(emb_dim + self.gnn_edge_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, return_attention_weights=False):
        """
        x: [N, emb_dim]
        edge_index: [2, E]
        edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        node_feature_extra [N, dim]
        """

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype).to(edge_vec.device)
        self_edge_vec[:, self.n_etype - 1] = 1

        head_type = node_type[edge_index[0]]  # [E, ] #head=src
        tail_type = node_type[edge_index[1]]  # [E, ] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]

        # print(torch.cat([edge_vec, headtail_vec], dim=1).shape)
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]
        # print(edge_embeddings.shape)

        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        # origin
        # x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)

        aggr_out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        assert len(edge_attr.size()) == 2
        # assert edge_attr.size(1) == self.emb_dim
        # assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(1) == x_j.size(1) == self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        # edge_attr:[200, 1024] x_i:[200, 4096] -> after cat:[200, 5120]
        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).reshape(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        if self.args.fp16 and self.training and self.args.upcast:
            with torch.cuda.amp.autocast(enabled=False):
                query = query.float() / math.sqrt(self.dim_per_head)
                scores = (query * key.float()).sum(dim=2)  # [E, heads]
        else:
            query = query / math.sqrt(self.dim_per_head)
            scores = (query * key).sum(dim=2)  # [E, heads]

        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]



class GAT(nn.Module):
    def __init__(self,
                 args,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 n_ntype,
                 n_etype,
                 heads,
                 gnn_layers = 3
                 ):
        super().__init__()
        self.gnn_layers = gnn_layers
        if args.use_relational_gnn:
            assert gnn_layers >= 2
            self.middle_conv_list = nn.ModuleList(
                [GATConvE(args, hidden_channels, n_ntype, n_etype) for _ in range(gnn_layers-1)]
            )
            self.end_conv = GATConvE(args, hidden_channels, n_ntype, n_etype)

        else:
            self.start_conv = GATConv(in_channels, hidden_channels, heads, dropout=0.6)

            if gnn_layers >= 3:
                self.middle_conv_list = nn.ModuleList([GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.6) for i in range(gnn_layers-2)])

            self.end_conv = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, args, x, edge_index, edge_type, node_type):
        num_nodes = node_type.size(0)
        invalid_src_idx = edge_index[0][edge_index[0] >= num_nodes]
        invalid_tgt_idx = edge_index[1][edge_index[1] >= num_nodes]
        # negative_idx = edge_index[(edge_index < 0).any(dim=0)]
        negative_mask = (edge_index < 0).any(dim=0)
        negative_idx = edge_index[:, negative_mask]

        if args.use_relational_gnn:
            x = F.dropout(x, p=0.6, training=self.training)
            for middle_conv in self.middle_conv_list:
                x = middle_conv(x, edge_index, edge_type, node_type)
                x= F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
            x = self.end_conv(x, edge_index, edge_type, node_type)

        else:
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.start_conv(x, edge_index)

            if self.gnn_layers >= 3:
                for middle_conv in self.middle_conv_list:
                    x = F.elu(x)
                    x = F.dropout(x, p=0.6, training=self.training)
                    x = middle_conv(x, edge_index)

            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.end_conv(x, edge_index)

        return x

