import pickle
import networkx as nx
from collections import defaultdict


def build_global_mapping(umls_graph):
    """
    构建全局统一的 sem_type_id→全局索引、rel_id→全局索引映射表
    :param umls_graph: 整个UMLS NetworkX图（已加载的对象）
    :return: sem_type2global_idx, rel_id2global_idx
    """
    # 全局实体类型映射（sem_type_id → 全局索引）
    sem_type2global_idx = defaultdict(lambda: len(sem_type2global_idx))
    # 全局关系类型映射（rel_id → 全局索引）
    rel_id2global_idx = defaultdict(lambda: len(rel_id2global_idx))

    # 遍历所有节点，收集sem_type_id（添加异常处理，避免属性缺失报错）
    for node in umls_graph.nodes():
        try:
            sem_type_id = umls_graph.nodes[node]["sem_type_id"]
            sem_type2global_idx[sem_type_id]  # 触发defaultdict自动分配索引
        except KeyError:
            print(f"警告：节点{node}缺少'sem_type_id'属性，跳过")
            continue

    # 遍历所有边，收集rel_id（添加异常处理，避免属性缺失报错）
    for u, v, attrs in umls_graph.edges(data=True):
        try:
            rel_id = attrs["rel_id"]
            rel_id2global_idx[rel_id]  # 触发defaultdict自动分配索引
        except KeyError:
            print(f"警告：边({u},{v})缺少'rel_id'属性，跳过")
            continue

    # 保存全局映射表（转为普通dict，避免defaultdict序列化问题）
    with open("global_mapping.pkl", "wb") as f:
        pickle.dump({
            "sem_type2global_idx": dict(sem_type2global_idx),  # 转为dict保存
            "rel_id2global_idx": dict(rel_id2global_idx)  # 转为dict保存
        }, f)

    # 打印全局统计结果（关键！用于后续模型初始化）
    global_n_ntype = len(sem_type2global_idx)
    global_n_etype = len(rel_id2global_idx)
    print(f"\n===== 全局类型统计结果 =====")
    print(f"全局实体类型数（n_ntype）：{global_n_ntype}")
    print(f"全局关系类型数（n_etype）：{global_n_etype}")
    print(f"全局映射表已保存到：global_mapping.pkl")

    return sem_type2global_idx, rel_id2global_idx

if __name__ == "__main__":
    umls_graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"  # 你的图文件路径（.pkl/.gpickle均可）

    with open(umls_graph_path, "rb") as f:
        umls_graph = pickle.load(f)

    # 方法2：如果图是用 NetworkX 的 write_gexf 保存的（.gexf后缀）
    # umls_graph = nx.read_gexf(umls_graph_path)

    # 验证是否为 NetworkX 图（避免加载错误）
    if not isinstance(umls_graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("加载的文件不是 NetworkX 图格式！请检查文件路径和格式")

    # 构建全局映射表
    build_global_mapping(umls_graph)

