import pandas as pd
import networkx as nx
import pickle
import vaex
from tqdm import tqdm
import os


def parse_umls_rrf_to_graph(umls_rrf_dir, output_graph_path, filter_source="MSH"):
    """
    兼容 NetworkX 2.x/3.x + 只保留英语数据 + vaex提速 + 新增MRDEF详细概念
    核心修改：1. 解析MRDEF.RRF提取详细定义 2. 将定义合并到节点属性
    """
    # --------------------------
    # 1. 解析MRCONSO.RRF（只保留英语，LAT='ENG'）
    # --------------------------
    print("正在解析MRCONSO.RRF（只保留英语概念）...")
    mrconso_path = os.path.join(umls_rrf_dir, "MRCONSO.RRF")

    conso_cols = ["CUI", "LAT", "STR"]
    conso_col_indices = [0, 1, 14]  # 0=CUI，1=LAT（语言），14=STR

    conso_df = pd.read_csv(
        mrconso_path,
        sep="|",
        header=None,
        usecols=conso_col_indices,
        names=conso_cols,
        encoding="latin-1",
        skiprows=1,
        low_memory=False
    )

    # 英语过滤+去重+空值过滤
    conso_df = conso_df[conso_df["LAT"] == "ENG"].reset_index(drop=True)
    conso_df = conso_df.drop_duplicates(subset="CUI", keep="first").reset_index(drop=True)
    conso_df = conso_df[conso_df["STR"].notna() & (conso_df["STR"] != "")].reset_index(drop=True)
    print(f"MRCONSO解析完成：{len(conso_df)} 个英语唯一概念节点\n")

    # --------------------------
    # 2. 解析MRSTY.RRF（语义类型信息）
    # --------------------------
    print("正在解析MRSTY.RRF（语义类型信息）...")
    mrsty_path = os.path.join(umls_rrf_dir, "MRSTY.RRF")

    sty_cols = ["CUI", "TUI", "STY"]
    sty_col_indices = [0, 1, 3]

    sty_df = pd.read_csv(
        mrsty_path,
        sep="|",
        header=None,
        usecols=sty_col_indices,
        names=sty_cols,
        encoding="latin-1",
        skiprows=1
    )
    sty_df = sty_df.drop_duplicates(subset="CUI", keep="first").reset_index(drop=True)

    # 只保留英语概念的语义类型
    english_cuis = set(conso_df["CUI"])
    sty_df = sty_df[sty_df["CUI"].isin(english_cuis)].reset_index(drop=True)
    print(f"MRSTY解析完成：{len(sty_df)} 个英语概念的语义类型\n")

    # --------------------------
    # 新增：3. 解析MRDEF.RRF（详细概念定义）
    # --------------------------
    print("正在解析MRDEF.RRF（详细概念定义）...")
    mrdef_path = os.path.join(umls_rrf_dir, "MRDEF.RRF")

    # MRDEF.RRF结构：第0列=CUI，第5列=DEF（详细定义），第4列=LAT（语言）
    def_cols = ["CUI", "LAT", "DEF"]
    def_col_indices = [0, 4, 5]  # 0=CUI，4=LAT（语言），5=DEF（详细定义）

    def_df = pd.read_csv(
        mrdef_path,
        sep="|",
        header=None,
        usecols=def_col_indices,
        names=def_cols,
        encoding="latin-1",
        skiprows=1,
        low_memory=False
    )

    # 过滤：英语定义 + 属于英语概念集 + 去重（一个CUI可能多个定义，保留第一个）
    def_df = def_df[
        (def_df["LAT"] == "ENG") &
        (def_df["CUI"].isin(english_cuis)) &
        (def_df["DEF"].notna()) & (def_df["DEF"] != "")
    ].drop_duplicates(subset="CUI", keep="first").reset_index(drop=True)

    # 构建CUI到定义的映射（方便后续合并）
    cui2def = dict(zip(def_df["CUI"], def_df["DEF"].str.strip()))
    print(f"MRDEF解析完成：{len(cui2def)} 个英语概念的详细定义\n")

    # --------------------------
    # 4. 合并节点属性（CUI+名称+语义类型+详细定义）
    # --------------------------
    print("合并节点属性...")
    node_df = pd.merge(
        conso_df[["CUI", "STR"]],
        sty_df,
        on="CUI",
        how="left"
    ).fillna({
        "TUI": "UNKNOWN_TUI",
        "STY": "Unknown Semantic Type"
    })

    # 添加详细定义（无定义则填充默认值）
    node_df["DEF"] = node_df["CUI"].map(cui2def).fillna("No detailed definition available.")

    node_attrs = {}
    for _, row in tqdm(node_df.iterrows(), total=len(node_df), desc="处理节点属性"):
        node_attrs[row["CUI"]] = {
            "cui": row["CUI"].strip(),
            "name": row["STR"].strip(),
            "sem_type_id": row["TUI"].strip(),
            "sem_type_text": row["STY"].strip(),
            "definition": row["DEF"].strip(),  # 新增：详细概念定义
            "language": "ENG"
        }
    print(f"节点属性合并完成：{len(node_attrs)} 个带完整属性的英语节点（含详细定义）\n")

    # --------------------------
    # 5. 解析MRREL.RRF（关系边）
    # --------------------------
    print("正在解析MRREL.RRF")
    mrrel_path = os.path.join(umls_rrf_dir, "MRREL.RRF")

    rel_cols = ["CUI1", "REL", "CUI2", "RELA", "SAB"]
    rel_col_indices = [0, 3, 4, 7, 10]
    valid_cuis = english_cuis

    # vaex读取+过滤
    df_rel = vaex.read_csv(
        mrrel_path,
        sep="|",
        header=None,
        usecols=rel_col_indices,
        names=rel_cols,
        encoding="latin-1",
        skiprows=1,
        dtype={
            "RELA": str,
            "REL": str,
            "SAB": str,
            "CUI1": str,
            "CUI2": str
        }
    )

    df_rel = df_rel[
        (df_rel["SAB"] == filter_source) &
        (df_rel["CUI1"].isin(valid_cuis)) &
        (df_rel["CUI2"].isin(valid_cuis)) &
        (df_rel["RELA"].notna()) & (df_rel["RELA"] != "") &
        (df_rel["REL"].notna()) & (df_rel["REL"] != "")
        ]

    rel_df = df_rel.to_pandas_df()
    print(f"MRREL解析完成：{len(rel_df)} 条英语有效关系边\n")

    # --------------------------
    # 6. 构建图谱并保存
    # --------------------------
    print("正在构建英语知识图谱...")
    G = nx.MultiDiGraph()

    # 添加节点（含详细定义）
    for cui, attrs in tqdm(node_attrs.items(), desc="添加节点"):
        G.add_node(cui, **attrs)

    # 添加边（去重）
    edge_attrs_set = set()
    for _, row in tqdm(rel_df.iterrows(), total=len(rel_df), desc="添加边"):
        edge_key = (row["CUI1"], row["CUI2"], row["RELA"].strip())
        if edge_key in edge_attrs_set:
            continue
        G.add_edge(
            row["CUI1"], row["CUI2"],
            rel_id=row["REL"].strip(),
            rela_text=row["RELA"].strip(),
            source=row["SAB"].strip()
        )
        edge_attrs_set.add(edge_key)

    with open(output_graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"\n英语UMLS图谱构建完成！")
    print(f"输出路径：{output_graph_path}")
    print(f"图谱统计：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")


# 运行函数（修改为你的实际路径！）
if __name__ == "__main__":
    UMLS_RRF_DIR = r"C:\Users\30975\PycharmProjects\pythonProject1\GraphNeuralPromter\dataset\umls\META"
    OUTPUT_GRAPH_PATH = r"C:\Users\30975\PycharmProjects\pythonProject1\GraphNeuralPromter\dataset\umls\processed_umls.pkl"  # 建议添加.pkl后缀
    parse_umls_rrf_to_graph(
        umls_rrf_dir=UMLS_RRF_DIR,
        output_graph_path=OUTPUT_GRAPH_PATH,
        filter_source="MSH"
    )
