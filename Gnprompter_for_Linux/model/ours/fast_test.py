import pickle
import re
import string
import time
from collections import defaultdict

from rapidfuzz import fuzz

from utils import batch_get_2hop_subgraph, \
    map_keywords_to_cuis_new

# 验证是否加载C++后端
print("C++后端是否生效:", hasattr(fuzz, '_fuzz_cpp'))  # 输出True才对


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

    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            index = data["index"]
            all_entities = data["all_entities"]
    except Exception as e:
        print(f"加载索引失败：{e}")
        return []

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

    return matched_entities





if __name__ == "__main__":
    # 1. 首次运行：构建UMLS索引（只需要执行一次！）
    umls_graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"  # 你的UMLS图谱路径
    
    # 检查是否已有索引文件，避免重复构建
    import os
    index_path = "umls_entity_index_1500000.pkl"
    if not os.path.exists(index_path):
        print("正在构建UMLS索引...")
        index_path = build_umls_index(umls_graph_path)
    else:
        print(f"使用已存在的索引文件: {index_path}")

    # 2. 准备测试问题集
    batch_statements = [
        "Is cytokeratin immunoreactivity useful in the diagnosis of short-segment Barrett's oesophagus in Korea?",
        "Is extended aortic replacement in acute type A dissection justifiable?",
        "Is double-balloon enteroscopy an accurate method to diagnose small-bowel disorders?",
        "Does tranexamic acid reduce desmopressin-induced hyperfibrinolysis?",
        "Should ascitis volume and anthropometric measurements be estimated in hospitalized alcoholic cirrotics?",
        "Emergency double - balloon enteroscopy combined with real-time viewing of capsule endoscopy: afeasible combined approach in acute overt - obscure gastrointestinal bleeding?"
    ]
    
    # 3. 测试单条问题的实体提取性能
    question = "What are the therapeutic drugs for non-small cell lung cancer?"
    print(f"\n测试单条问题: {question}")
    start_time = time.time()
    matched = fast_extract_matched_entities(question, index_path=index_path)
    end_time = time.time()
    print(f"实体提取完成，耗时: {(end_time - start_time)*1000:.2f}毫秒")
    print(f"匹配结果数量: {len(matched)}")
    print(f"匹配结果前10个: {matched[:10]}")
    
    # 4. 批量测试实体提取性能
    print(f"\n批量测试 {len(batch_statements)} 条问题的实体提取性能...")
    total_time = 0
    total_matched_entities = 0
    all_entities = []
    
    for i, statement in enumerate(batch_statements):
        print(f"\n问题 {i+1}: {statement}")
        batch_start_time = time.time()
        batch_matched = fast_extract_matched_entities(statement, index_path=index_path)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_time += batch_time
        total_matched_entities += len(batch_matched)

        all_entities.append(list(batch_matched))
        
        print(f"  实体提取耗时: {batch_time*1000:.2f}毫秒")
        print(f"  匹配到的实体数: {len(batch_matched)}")
        print(f"  匹配到的实体前5个: {batch_matched[:5]}")
    
    # 5. 性能统计汇总
    print("\n=== 实体提取性能统计汇总 ===")
    print(f"总测试问题数: {len(batch_statements)}")
    print(f"总耗时: {total_time*1000:.2f}毫秒")
    print(f"平均每个问题提取耗时: {(total_time/len(batch_statements))*1000:.2f}毫秒")
    print(f"平均每个问题匹配到的实体数: {total_matched_entities/len(batch_statements):.2f}")
    print(f"总匹配实体数: {total_matched_entities}")
    print(all_entities)
    print("=" * 100)

    graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"

    with open(graph_path, "rb") as f:
        full_graph = pickle.load(f)
    cui2name_full = {cui: full_graph.nodes[cui]["name"] for cui in full_graph.nodes()}

    name_to_cuis = {}
    for cui, node_name in cui2name_full.items():
        if node_name:  # 过滤空名称的节点
            name_lower = node_name.lower()
            if name_lower not in name_to_cuis:
                name_to_cuis[name_lower] = []
            name_to_cuis[name_lower].append(cui)

    new_cuis = map_keywords_to_cuis_new(all_entities, name_to_cuis)

    print(new_cuis)
    batch_subgraphs = batch_get_2hop_subgraph(full_graph, new_cuis, max_nodes=500)
    # print(batch_subgraphs[0].number_of_nodes())
    # print(batch_subgraphs[0].number_of_edges())
    # print(batch_subgraphs[1].number_of_nodes())
    # print(batch_subgraphs[1].number_of_edges())
    # print(batch_subgraphs[2].number_of_nodes())
    # print(batch_subgraphs[2].number_of_edges())




