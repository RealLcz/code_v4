import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from collections import defaultdict

# generate the neg and pos triples for RotatEDcoder
def generate_pos_neg_triples(triples, n_nodes, neg_sample_size=5):
    """

    :param triples: original triples [E, 3] (head_idx, relation_idx, tail_idx)
    :param n_nodes:
    :param neg_sample_size:
    :return: pos_triples( 3 * [B, ] tensor), neg_triples(3 * [B,K] tensor)
    """
    # positive triples
    sample_size = min(10000, len(triples))  # avoid the size is too big
    pos_idx = torch.randperm(len(triples))[:sample_size]
    pos_triples = (
        triples[pos_idx, 0],
        triples[pos_idx, 1],
        triples[pos_idx, 2]
    )

    # negative triples
    B = sample_size
    K = neg_sample_size
    neg_tail = torch.randint(0, n_nodes, (B, K), dtype=torch.long)
    neg_triples = (
        pos_triples[0].unsqueeze(1).repeat(1, K),
        pos_triples[1].unsqueeze(1).repeat(1, K),
        neg_tail
    )

    return pos_triples, neg_triples


import pickle


def umls_graph_to_pyg_data_subgraph(args, subgraph, tokenizer, model, project_layer):
    """
    将2-hop子图转换为PyG数据（使用全局映射表，避免局部索引越界）
    修复点：edge_index格式、动态修改全局映射、重复创建投影层、冗余代码清理
    """
    # 1. 加载全局映射表（提前构建好的，只读不修改）
    with open("global_mapping.pkl", "rb") as f:
        global_mapping = pickle.load(f)
    sem_type2global_idx = global_mapping["sem_type2global_idx"]
    rel_id2global_idx = global_mapping["rel_id2global_idx"]

    # 2. 全局类型数（从映射表获取，固定不变）
    global_n_ntype = len(sem_type2global_idx)
    global_n_etype = len(rel_id2global_idx)

    # 3. 提取子图的边和节点
    edges = list(subgraph.edges(data=True))
    nodes = list(subgraph.nodes(data=True))
    cui_list = list(subgraph.nodes())  # 子图节点CUI列表
    n_nodes = len(cui_list)

    # 4. 子图节点CUI→连续ID（子图内部的局部ID，用于边索引）
    cui2idx = {cui: idx for idx, cui in enumerate(cui_list)}

    # 5. 处理边索引和边类型（严格遵循PyG [2, E] 格式）
    edge_index = []
    edge_rel_ids = []
    for u, v, attrs in subgraph.edges(data=True):
        u_idx = cui2idx[u]
        v_idx = cui2idx[v]
        edge_index.append([u_idx, v_idx])  # 临时存储格式：[E, 2]
        edge_rel_ids.append(attrs["rel_id"])

    # 关键修复：确保edge_index是 [2, E] 格式（PyG标准）
    if len(edge_index) == 0:
        # 无边子图：创建 [2, 0] 空张量（维度正确，无无效边）
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # 有边子图：转置为 [2, E] 并确保内存连续
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    print(edge_index.shape)
    # 6. 节点类型（使用全局映射，禁止动态修改映射表）
    sem_type_ids = [subgraph.nodes[cui]["sem_type_id"] for cui in cui_list]
    node_type = []
    for st in sem_type_ids:
        if st not in sem_type2global_idx:
            # 发现未收录类型：报错提示更新全局映射表（避免数据不一致）
            raise ValueError(
                f"子图中发现未收录的sem_type_id: {st}\n"
                f"请更新 global_mapping.pkl 以包含该类型"
            )
        node_type.append(sem_type2global_idx[st])
    node_type = torch.tensor(node_type, dtype=torch.long)

    # 7. 边类型（使用全局映射，禁止动态修改映射表）
    edge_type = []
    for rid in edge_rel_ids:
        if rid not in rel_id2global_idx:
            # 发现未收录关系：报错提示更新全局映射表
            raise ValueError(
                f"子图中发现未收录的rel_id: {rid}\n"
                f"请更新 global_mapping.pkl 以包含该关系"
            )
        edge_type.append(rel_id2global_idx[rid])
    # 确保edge_type与edge_index边数一致（无边时为空张量）
    edge_type = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty(0, dtype=torch.long)

    # 8. 文本属性（保持原有逻辑）
    text_attributes = [subgraph.nodes[cui]["definition"] for cui in cui_list]

    # 9. 生成512维节点嵌入（使用外部传入的投影层，避免重复创建）
    inputs = tokenizer(
        text_attributes,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=32
    ).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        cls_embeds = outputs.hidden_states[-1][:, 0, :]  # [N, 4096]
        node_embeds = project_layer(cls_embeds)  # [N, 512]

    # 10. 构建PyG数据（字段完整，格式符合PyG要求）
    pyg_data = Data(
        x=node_embeds,               # 节点嵌入 [N, 512]
        edge_index=edge_index,       # 边索引 [2, E]
        edge_type=edge_type,         # 边类型 [E]
        node_type=node_type,         # 节点类型 [N]
        num_nodes=n_nodes,           # 节点数
        n_ntype=global_n_ntype,      # 全局节点类型数
        n_etype=global_n_etype+1       # 全局边类型数
    )

    # 可选：保留triples（若后续需要使用，否则可删除）
    triples = []
    if edge_index.size(1) > 0:  # 有边时才构建三元组
        for u_idx, v_idx, rel_idx in zip(edge_index[0], edge_index[1], edge_type):
            triples.append((u_idx.item(), rel_idx.item(), v_idx.item()))
    triples = torch.tensor(triples, dtype=torch.long) if triples else torch.empty((0, 3), dtype=torch.long)

    return pyg_data, text_attributes, triples, global_n_ntype, global_n_etype + 1


def batch_umls_graph_to_pyg_data(batch_subgraphs, args, device, tokenizer, model, project_layer):
    """
    批量子图转PyG数据（恢复随机选节点，确保类型索引合法，避免CUDA报错）
    """
    B = len(batch_subgraphs)
    N = max(subgraph.number_of_nodes() for subgraph in batch_subgraphs)  # 子图最大实体数
    gnn_in_dim = args.gnn_in_dim

    # 初始化批量存储
    batch_gat_inputs = []
    batch_pyg_graphs = []
    batch_text_attrs = []
    batch_triples_list = []
    offset = 0  # 实体索引偏移

    n_ntype = 127
    n_etype = 11

    for i, subgraph in enumerate(batch_subgraphs):
        # 单个子图转PyG
        pyg_data, text_attrs, triples, sub_n_ntype, sub_n_etype = umls_graph_to_pyg_data_subgraph(args, subgraph, tokenizer, model, project_layer)

        # 核心修复1：裁剪node_type到合法范围（0 ≤ node_type < n_ntype）
        if hasattr(pyg_data, 'node_type'):
            # 裁剪无效值（-1→0，≥128→127）
            pyg_data.node_type = torch.clamp(pyg_data.node_type, 0, n_ntype - 1)
        else:
            # 无node_type时，默认赋值为0（合法索引）
            pyg_data.node_type = torch.zeros(subgraph.number_of_nodes(), dtype=torch.long, device=device)

        # 核心修复2：裁剪edge_type到合法范围（0 ≤ edge_type < n_etype）
        if hasattr(pyg_data, 'edge_type'):
            pyg_data.edge_type = torch.clamp(pyg_data.edge_type, 0, n_etype)
        else:
            # 无边时edge_type为空，或自环边设为0
            pyg_data.edge_type = torch.tensor([], dtype=torch.long, device=device)

        # 实体特征padding到N
        pad_nodes = N - pyg_data.num_nodes
        if pad_nodes > 0:
            pyg_data.x = F.pad(pyg_data.x, (0, 0, 0, pad_nodes))  # [N, gnn_in_dim]
            pyg_data.num_nodes = N
            text_attrs += ["<PAD>"] * pad_nodes
            # padding部分的node_type设为0（合法索引）
            pyg_data.node_type = F.pad(pyg_data.node_type, (0, pad_nodes), value=0)

        # 存储数据
        batch_gat_inputs.append(pyg_data.x)
        batch_pyg_graphs.append(pyg_data)
        batch_text_attrs.append(text_attrs)
        batch_triples_list.append(triples)

        # 打印日志
        if subgraph.number_of_edges() == 0:
            print(f"警告：第{i}个子图无边缘，节点数：{subgraph.number_of_nodes()}")

    # 拼接批量数据
    gat_inputs = torch.cat(batch_gat_inputs, dim=0).to(device)
    batch_text_attributes = batch_text_attrs

    # 三元组padding（兼容空三元组）
    max_K = max(t.size(0) for t in batch_triples_list) if batch_triples_list else 1
    batch_triples = []
    for triples in batch_triples_list:
        if len(triples.shape) == 1:
            triples = triples.view(-1, 3)
        if triples.dim() != 2:
            raise ValueError(f"三元组维度错误: 预期2维，实际{triples.dim()}维")
        pad_K = max_K - triples.size(0)
        if pad_K > 0:
            triples = F.pad(triples, (0, 0, 0, pad_K), value=-1)
        batch_triples.append(triples.to(device))

    return gat_inputs, batch_pyg_graphs, batch_text_attributes, batch_triples
