import pickle
import re
import string
from collections import defaultdict

import networkx as nx
from rapidfuzz import fuzz
from tqdm import tqdm


def simple_tokenize(text):
    """简单分词：按空格、连字符、斜杠分割，保留有效词"""
    # 把特殊分隔符替换为空格，再按空格分割
    text = re.sub(r'[-/]', ' ', text)  # 把 "-" 和 "/" 换成空格
    words = text.split()  # 按空格分割
    return [word for word in words if len(word) >= 8]  # 过滤过短词

def build_umls_index(umls_graph_path, index_save_path="umls_entity_index_1500000.pkl"):
    with open(umls_graph_path, "rb") as f:
        G = pickle.load(f)

    def simple_tokenize(text):
        """简单分词：按空格、连字符、斜杠分割，保留有效词"""
        # 把特殊分隔符替换为空格，再按空格分割
        text = re.sub(r'[-/]', ' ', text)  # 把 "-" 和 "/" 换成空格
        words = text.split()  # 按空格分割
        return [word for word in words if len(word) >= 3]  # 过滤过短词

    umls_entities = set()
    for _, attrs in G.nodes(data=True):
        if isinstance(attrs, dict) and "name" in attrs and attrs["name"]:
            name = attrs["name"].strip().lower()
            if len(name) >= 8:
                umls_entities.add(name)

    umls_entities = list(umls_entities)
    print(f"number of umls entities:{len(umls_entities)}")

    index = defaultdict(list)

    # def get_ngrams(s, n):
    #     return [s[i:i + n] for i in range(len(s) - n + 1)]

    for entity in umls_entities:
        words = simple_tokenize(entity)
        for word in words:
            if len(word) >= 4:
                index[word].append(entity)

        # ngrams = get_ngrams(entity, 8)
        # for gram in ngrams:
        #     index[gram].append(entity)

        # 3. 索引去重（同一关键词下的实体不重复）
    for key in index:
        index[key] = list(set(index[key]))  # 去重，减少候选集大小

        # 4. 保存索引（后续匹配直接加载，无需重复构建）
    with open(index_save_path, "wb") as f:
        pickle.dump({"index": index, "all_entities": umls_entities}, f)
    print(f"索引构建完成，保存路径：{index_save_path}")
    return index_save_path

def fast_extract_matched_entities(question_text, index_path="umls_entity_index_1500000.pkl", fuzzy_threshold=88):
    """
    快速实体匹配：先通过索引筛选候选，再模糊匹配（毫秒级）
    """
    if not question_text or not index_path:
        return []

    # 1. 加载索引（比加载整个UMLS图谱快10倍+）
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            index = data["index"]
            all_entities = data["all_entities"]
    except Exception as e:
        print(f"加载索引失败：{e}")
        return []

    # 2. 问题文本预处理（和原逻辑一致，但更简洁）
    allowed_chars = set(string.ascii_lowercase + string.digits + " -/")
    clean_text = "".join([c if c in allowed_chars or c.isspace() else " " for c in question_text.lower()])
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = f" {clean_text} "

    # 3. 提取问题的关键词/ngram（用于查询索引）
    query_keys = set()
    # （1）问题分词，提取关键词
    question_words = simple_tokenize(clean_text)
    for word in question_words:
        if len(word) >= 8:
            query_keys.add(word)
    # （2）问题提取ngram，覆盖部分匹配
    def get_ngrams(s, n):
        return [s[i:i+n] for i in range(len(s)-n+1)]
    query_ngrams = get_ngrams(clean_text.replace(" ", ""), 8)
    query_keys.update(query_ngrams)

    # 4. 索引查询：快速筛选候选实体（核心提速步骤）
    candidate_entities = set()  # 用set去重，避免重复候选
    for key in query_keys:
        if key in index:
            candidate_entities.update(index[key])  # 把索引中对应的实体加入候选集

    # 兜底：若候选集为空（极端情况），取前1000个实体做模糊匹配（避免漏匹配）
    if not candidate_entities:
        candidate_entities = set(all_entities[:1000])
    print(f"筛选后候选实体数：{len(candidate_entities)}（原实体数：{len(all_entities)}）")

    # 5. 候选集模糊匹配（只对候选集处理，大幅减少计算量）
    matched_entities = []
    seen_entities = set()

    for entity in candidate_entities:
        # 优先精准匹配（快速跳过，减少模糊计算）
        if f" {entity} " in clean_text:
            formatted = entity.title()
            if formatted not in seen_entities:
                seen_entities.add(formatted)
                matched_entities.append(formatted)
            continue

        # 模糊匹配（只对长度≥4的实体处理）
        if len(entity) >= 8:
            # 用partial_ratio（部分匹配），容忍轻微变异（如单复数、拼写误差）
            try:
                score = fuzz.partial_ratio(entity, clean_text)
                if score >= fuzzy_threshold:
                    formatted = entity.title()
                    if formatted not in seen_entities:
                        seen_entities.add(formatted)
                        matched_entities.append(formatted)
            except Exception as e:
                continue

    if len(matched_entities) == 0:
        print(f"没有匹配到实体")

    print(matched_entities)

    return matched_entities


def map_keywords_to_cuis_new(batch_keywords, name_to_cuis):
    """
    批量将关键词列表映射到UMLS的CUI（仅精确匹配，保持batch结构，优先保证速度）
    :param batch_keywords: 批量关键词列表，格式如 [['kw1', 'kw2'], ['kw3'], ...]（共6个列表）
    :param name_to_cuis: 预构建的反向映射（名称小写→CUI列表），格式如 {'diabetes': ['C0011849']}
    :return: 批量CUI列表，与输入batch结构一致，空关键词列表对应空CUI列表
    """
    batch_cuis = []

    # 批量处理每个样本的关键词（避免嵌套循环冗余，提升速度）
    for keywords in batch_keywords:
        # 单个样本的CUI结果（临时存储）
        sample_cuis = []

        # 关键词预处理（小写+去空格+过滤无效值），单次循环完成，无额外开销
        for kw in keywords:
            if not kw:
                continue
            processed_kw = kw.lower().strip()
            if len(processed_kw) < 2:
                continue

            # 精确匹配（字典查找O(1)，速度最快）
            if processed_kw in name_to_cuis:
                # 直接扩展CUI（假设name_to_cuis中CUI已为字符串，无列表嵌套）
                sample_cuis.extend(name_to_cuis[processed_kw])

        # 去重（用dict.fromkeys保持顺序，比set+list快且有序，Python3.7+）
        sample_cuis = list(dict.fromkeys(sample_cuis))

        # 加入批量结果，保持与输入batch的结构一致
        batch_cuis.append(sample_cuis)

    return batch_cuis


def get_2hop_subgraph(G, matched_cuis, max_nodes=50):
    """提取2跳子图，增强异常处理和边界检查，确保提取的子图有边"""
    try:
        # 输入验证
        if G is None:
            print("错误: 输入图G为None")
            return G
        if not isinstance(max_nodes, int) or max_nodes <= 0:
            max_nodes = 50
            print("警告: max_nodes无效，使用默认值50")

        # 确保matched_cuis不为空，如果为空，从图中随机选择一些节点
        valid_cuis = [cui for cui in matched_cuis if cui and cui in G]
        if not valid_cuis and len(G) > 0:
            # 如果没有匹配的CUI，从图中随机选择5个节点作为种子
            import random
            valid_cuis = random.sample(list(G.nodes()), min(5, len(G)))
            print(f"没有匹配的CUI，随机选择了{len(valid_cuis)}个节点作为种子")
        elif not valid_cuis:
            return G.subgraph([]).copy()  # 图为空的情况

        # 优化1: 一次性处理所有种子节点，避免多次BFS
        visited = set(valid_cuis)
        queue = list(valid_cuis)

        # 优化2: 移除嵌套的tqdm进度条，使用单个tqdm显示整体进度
        hops = 0
        while queue and len(visited) < max_nodes and hops < 2:
            new_queue = []
            for node in queue:
                if G.has_node(node):
                    neighbors = list(G.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor in G:
                            visited.add(neighbor)
                            new_queue.append(neighbor)
                        # 提前终止条件
                        if len(visited) >= max_nodes:
                            break
                    if len(visited) >= max_nodes:
                        break
            queue = new_queue
            hops += 1

        # 使用visited作为子图节点
        subgraph_nodes = visited

        # 如果子图节点数太少，扩展到更多邻居
        if len(subgraph_nodes) < 10 and len(G) > 0:
            # 扩展现有节点的邻居
            additional_nodes = set()
            for node in list(subgraph_nodes):
                if G.has_node(node):
                    neighbors = list(G.neighbors(node))
                    additional_nodes.update([n for n in neighbors if n in G])
            subgraph_nodes.update(additional_nodes)
            # 再次控制大小
            if len(subgraph_nodes) > max_nodes:
                subgraph_nodes = list(subgraph_nodes)[:max_nodes]

        # 安全创建子图
        try:
            subgraph = G.subgraph(subgraph_nodes).copy()

            # 检查子图是否有边，如果没有，添加一些额外的节点
            if len(subgraph.edges()) == 0 and len(subgraph_nodes) > 0:
                print(f"警告: 提取的子图没有边，尝试扩展节点集")
                # 从原始图中找到与现有节点相关的更多节点
                extended_nodes = set(subgraph_nodes)
                # for node in subgraph_nodes:
                #     if G.has_node(node):
                #         # 最多添加10个邻居
                #         neighbors = list(G.neighbors(node))[:10] if G.has_node(node) else []
                #         extended_nodes.update([n for n in neighbors if n in G])
                #
                # # 创建扩展后的子图
                subgraph = G.subgraph(extended_nodes).copy()

            return subgraph
        except Exception as e:
            print(f"创建子图时出错: {e}")
            return G.subgraph([]).copy()  # 返回空图
    except Exception as e:
        print(f"提取2跳子图时发生未预期错误: {e}")
        return G.subgraph([]).copy() if G else None


# data_processing/umls_utils.py（新增/修改）
def batch_get_2hop_subgraph(full_graph, batch_matched_cuis, max_nodes=500):
    """
    批量提取2跳子图，增强异常处理和边界检查
    :param full_graph: 完整UMLS图谱
    :param batch_matched_cuis: [B, K] → B个问题，每个问题匹配K个CUI
    :param max_nodes: 每个子图最大实体数
    :return: batch_subgraphs → [B] 列表，每个元素是子图
    """
    try:
        # 输入验证
        if full_graph is None:
            print("错误: 输入图谱full_graph为None")
            return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]
        if not isinstance(batch_matched_cuis, list):
            print("错误: batch_matched_cuis必须是列表类型")
            return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]
        if not isinstance(max_nodes, int) or max_nodes <= 0:
            max_nodes = 500
            print("警告: max_nodes无效，使用默认值500")

        batch_subgraphs = []
        # 优化: 使用tqdm显示批量处理进度
        for i, cuis in enumerate(tqdm(batch_matched_cuis, desc="批量处理子图")):
            try:
                # 验证单批次CUI输入
                if cuis is None:
                    cuis = []
                elif not isinstance(cuis, (list, set)):
                    print(f"警告: 第{i}个批次的cuis必须是列表或集合类型，已跳过")
                    batch_subgraphs.append(full_graph.subgraph([]).copy())
                    continue

                # 提取子图
                subgraph = get_2hop_subgraph(full_graph, cuis, max_nodes)
                batch_subgraphs.append(subgraph)
            except Exception as e:
                print(f"处理第{i}个批次时出错: {e}")
                # 添加空图以保持批次数量一致性
                batch_subgraphs.append(full_graph.subgraph([]).copy())

        return batch_subgraphs
    except Exception as e:
        print(f"批量提取2跳子图时发生未预期错误: {e}")
        return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]


# if __name__ == "__main__":
#     # 1. 首次运行：构建UMLS索引（只需要执行一次！）
#     umls_graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"  # 你的UMLS图谱路径
#     index_path = build_umls_index(umls_graph_path)  # 生成索引文件
#
#     # 2. 后续匹配：直接用快速匹配函数（毫秒级）
#     question = "What are the therapeutic drugs for non-small cell lung cancer?"
#     matched = fast_extract_matched_entities(question, index_path=index_path)
#     print(f"匹配结果：{matched}")

