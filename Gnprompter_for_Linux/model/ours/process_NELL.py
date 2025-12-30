import pandas as pd
import networkx as nx
import pickle
import vaex
from tqdm import tqdm
import os


def parse_nell_to_graph(nell_csv_path, output_graph_path, min_probability=0.0, filter_sources=None):
    """
    修复vaex API兼容问题 + 适配NELL制表符格式 + 仅提取必需列
    核心修复：替换vaex.expression.and_为兼容写法，适配新旧版本vaex
    """
    # --------------------------
    # 1. 解析NELL主数据（制表符分隔，仅读必需列）
    # --------------------------
    print("正在解析NELL数据集（制表符分隔）...")

    # 必需列（精确匹配数据集表头）
    required_cols = [
        "Entity",
        "Relation",
        "Value",
        "Probability",
        "Source",
        "Entity literalStrings",
        "Value literalStrings"
    ]

    # 使用vaex读取（制表符分隔，仅加载必需列）
    try:
        df_nell = vaex.read_csv(
            nell_csv_path,
            sep="\t",  # 关键：制表符分隔
            header=0,
            usecols=required_cols,
            encoding="utf-8",
            dtype={
                "Entity": str,
                "Relation": str,
                "Value": str,
                "Probability": float,
                "Source": str,
                "Entity literalStrings": str,
                "Value literalStrings": str
            },
            skip_blank_lines=True,
            na_values=["", " ", "\t", "nan"]
        )
    except Exception as e:
        print(f"Vaex读取失败，降级为pandas读取：{e}")
        # 降级方案：pandas读取后转vaex
        temp_df = pd.read_csv(
            nell_csv_path,
            sep="\t",
            header=0,
            usecols=required_cols,
            encoding="utf-8",
            dtype={
                "Entity": str,
                "Relation": str,
                "Value": str,
                "Probability": float,
                "Source": str,
                "Entity literalStrings": str,
                "Value literalStrings": str
            },
            skip_blank_lines=True,
            na_values=["", " ", "\t"]
        )
        df_nell = vaex.from_pandas(temp_df)

    # --------------------------
    # 修复核心：兼容vaex布尔过滤（替代vaex.expression.and_）
    # --------------------------
    # 基础过滤条件（分步组合，兼容所有vaex版本）
    df_nell = df_nell[df_nell["Entity"].notna()]
    df_nell = df_nell[df_nell["Value"].notna()]
    df_nell = df_nell[df_nell["Relation"].notna()]
    df_nell = df_nell[df_nell["Probability"] >= min_probability]

    # 源过滤（如果指定）
    if filter_sources is not None and isinstance(filter_sources, list):
        df_nell = df_nell[df_nell["Source"].isin(filter_sources)]

    # 转换为pandas（仅包含必需列）
    nell_df = df_nell.to_pandas_df()
    # 清理字符串列的空白字符
    str_cols = ["Entity", "Relation", "Value", "Source", "Entity literalStrings", "Value literalStrings"]
    for col in str_cols:
        if col in nell_df.columns:
            nell_df[col] = nell_df[col].astype(str).str.strip()

    print(f"NELL解析完成：{len(nell_df)} 条有效关系记录\n")

    # --------------------------
    # 2. 提取所有节点（仅核心属性）
    # --------------------------
    print("提取节点属性...")

    # 提取Entity节点
    entity_nodes = nell_df[["Entity", "Entity literalStrings"]].drop_duplicates(subset="Entity", keep="first")
    entity_nodes.rename(columns={"Entity": "node_id", "Entity literalStrings": "literal_strings"}, inplace=True)
    entity_nodes["node_type"] = "Entity"

    # 提取Value节点
    value_nodes = nell_df[["Value", "Value literalStrings"]].drop_duplicates(subset="Value", keep="first")
    value_nodes.rename(columns={"Value": "node_id", "Value literalStrings": "literal_strings"}, inplace=True)
    value_nodes["node_type"] = "Value"

    # 合并节点并填充默认值
    all_nodes = pd.concat([entity_nodes, value_nodes], ignore_index=True).drop_duplicates(subset="node_id",
                                                                                          keep="first")
    all_nodes["literal_strings"] = all_nodes["literal_strings"].fillna("No literal strings available").str.strip()
    all_nodes = all_nodes[all_nodes["node_id"] != ""].reset_index(drop=True)

    # 构建节点属性字典
    node_attrs = {}
    for _, row in tqdm(all_nodes.iterrows(), total=len(all_nodes), desc="处理节点"):
        node_id = row["node_id"]
        node_attrs[node_id] = {
            "node_id": node_id,
            "node_type": row["node_type"],
            "language": "ENG",
            "literal_strings": row["literal_strings"]
        }

    print(f"节点处理完成：{len(node_attrs)} 个唯一节点\n")

    # --------------------------
    # 3. 提取关系边（仅核心属性）
    # --------------------------
    print("提取关系边...")

    # 仅保留核心边属性
    edge_df = nell_df[["Entity", "Relation", "Value", "Probability", "Source"]].copy()
    # 去重 + 过滤无效节点
    edge_df = edge_df.drop_duplicates(subset=["Entity", "Relation", "Value"], keep="first")
    edge_df = edge_df[
        edge_df["Entity"].isin(node_attrs.keys()) &
        edge_df["Value"].isin(node_attrs.keys())
        ].reset_index(drop=True)

    # 构建边列表（去重）
    edge_attrs_set = set()
    edges = []
    for _, row in tqdm(edge_df.iterrows(), total=len(edge_df), desc="处理边"):
        src = row["Entity"]
        dst = row["Value"]
        rel = row["Relation"]

        # 防止重复边
        edge_key = (src, dst, rel)
        if edge_key in edge_attrs_set:
            continue

        edges.append((
            src, dst,
            {
                "relation_type": rel,
                "probability": round(row["Probability"], 4),
                "source": row["Source"] if row["Source"] != "" else "Unknown Source"
            }
        ))
        edge_attrs_set.add(edge_key)

    print(f"边处理完成：{len(edges)} 条唯一关系边\n")

    # --------------------------
    # 4. 构建图谱并保存
    # --------------------------
    print("构建NELL知识图谱...")
    G = nx.MultiDiGraph()

    # 批量添加节点（高效）
    G.add_nodes_from([(node_id, attrs) for node_id, attrs in node_attrs.items()])

    # 批量添加边（高效）
    G.add_edges_from([(src, dst, attrs) for src, dst, attrs in edges])

    # 保存图谱
    with open(output_graph_path, "wb") as f:
        pickle.dump(G, f)

    # 输出统计信息
    print(f"\n✅ NELL图谱构建完成！")
    print(f"📁 输出路径：{output_graph_path}")
    print(f"📊 统计信息：")
    print(f"   - 总节点数：{G.number_of_nodes()}")
    print(f"   - 总边数：{G.number_of_edges()}")
    print(f"   - 唯一关系类型数：{len(set([d['relation_type'] for _, _, d in G.edges(data=True)]))}")
    print(f"   - Entity节点数：{sum(1 for _, d in G.nodes(data=True) if d['node_type'] == 'Entity')}")
    print(f"   - Value节点数：{sum(1 for _, d in G.nodes(data=True) if d['node_type'] == 'Value')}")


# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 配置参数（替换为你的路径）
    NELL_CSV_PATH = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\NELL\NELL.08m.1115.esv.csv"
    OUTPUT_GRAPH_PATH = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\NELL\processed_NELL"
    MIN_PROBABILITY = 0.8  # 过滤低置信度关系
    FILTER_SOURCES = None  # 可选：指定源列表，如["Wikipedia"]

    # 执行转换
    parse_nell_to_graph(
        nell_csv_path=NELL_CSV_PATH,
        output_graph_path=OUTPUT_GRAPH_PATH,
        min_probability=MIN_PROBABILITY,
        filter_sources=FILTER_SOURCES
    )