import pickle
import re
import string
from collections import defaultdict

import networkx as nx
from rapidfuzz import fuzz
from tqdm import tqdm


def simple_tokenize(text):
    text = re.sub(r'[-/]', ' ', text)
    words = text.split()
    return [word for word in words if len(word) >= 8]

def build_umls_index(umls_graph_path, index_save_path="umls_entity_index_1500000.pkl"):
    with open(umls_graph_path, "rb") as f:
        G = pickle.load(f)

    def simple_tokenize(text):
        text = re.sub(r'[-/]', ' ', text)
        words = text.split()
        return [word for word in words if len(word) >= 3]

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

    for key in index:
        index[key] = list(set(index[key]))

    with open(index_save_path, "wb") as f:
        pickle.dump({"index": index, "all_entities": umls_entities}, f)
    print(f"{index_save_path}")
    return index_save_path

def fast_extract_matched_entities(question_text, index_path="umls_entity_index_1500000.pkl", fuzzy_threshold=88):
    if not question_text or not index_path:
        return []

    # 1. 加载索引（比加载整个UMLS图谱快10倍+）
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            index = data["index"]
            all_entities = data["all_entities"]
    except Exception as e:
        print(f"{e}")
        return []

    allowed_chars = set(string.ascii_lowercase + string.digits + " -/")
    clean_text = "".join([c if c in allowed_chars or c.isspace() else " " for c in question_text.lower()])
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = f" {clean_text} "

    query_keys = set()
    # question_words = simple_tokenize(clean_text)
    # for word in question_words:
    #     if len(word) >= 8:
    #         query_keys.add(word)
    # def get_ngrams(s, n):
    #     return [s[i:i+n] for i in range(len(s)-n+1)]
    # query_ngrams = get_ngrams(clean_text.replace(" ", ""), 8)
    # query_keys.update(query_ngrams)

    candidate_entities = set()
    for key in query_keys:
        if key in index:
            candidate_entities.update(index[key])

    if not candidate_entities:
        candidate_entities = set(all_entities[:1000])


    matched_entities = []
    seen_entities = set()

    for entity in candidate_entities:
        if f" {entity} " in clean_text:
            formatted = entity.title()
            if formatted not in seen_entities:
                seen_entities.add(formatted)
                matched_entities.append(formatted)
            continue

        if len(entity) >= 8:
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


def map_keywords_to_cuis_new(batch_keywords, name_to_cuis):

    batch_cuis = []

    for keywords in batch_keywords:
        sample_cuis = []

        for kw in keywords:
            if not kw:
                continue
            processed_kw = kw.lower().strip()
            if len(processed_kw) < 2:
                continue

            if processed_kw in name_to_cuis:
                sample_cuis.extend(name_to_cuis[processed_kw])

        sample_cuis = list(dict.fromkeys(sample_cuis))

        batch_cuis.append(sample_cuis)

    return batch_cuis


def get_2hop_subgraph(G, matched_cuis, max_nodes=50):
    try:
        # 输入验证
        if G is None:
            print("error")
            return G
        if not isinstance(max_nodes, int) or max_nodes <= 0:
            max_nodes = 50
            print("no max_nodes")
        valid_cuis = [cui for cui in matched_cuis if cui and cui in G]
        if not valid_cuis and len(G) > 0:
            import random
            valid_cuis = random.sample(list(G.nodes()), min(5, len(G)))
        elif not valid_cuis:
            return G.subgraph([]).copy()

        visited = set(valid_cuis)
        queue = list(valid_cuis)

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
                        if len(visited) >= max_nodes:
                            break
                    if len(visited) >= max_nodes:
                        break
            queue = new_queue
            hops += 1

        subgraph_nodes = visited

        if len(subgraph_nodes) < 10 and len(G) > 0:
            additional_nodes = set()
            for node in list(subgraph_nodes):
                if G.has_node(node):
                    neighbors = list(G.neighbors(node))
                    additional_nodes.update([n for n in neighbors if n in G])
            subgraph_nodes.update(additional_nodes)
            if len(subgraph_nodes) > max_nodes:
                subgraph_nodes = list(subgraph_nodes)[:max_nodes]
        try:
            subgraph = G.subgraph(subgraph_nodes).copy()
            if len(subgraph.edges()) == 0 and len(subgraph_nodes) > 0:
                extended_nodes = set(subgraph_nodes)
                # for node in subgraph_nodes:
                #     if G.has_node(node):
                #         neighbors = list(G.neighbors(node))[:10] if G.has_node(node) else []
                #         extended_nodes.update([n for n in neighbors if n in G])
                subgraph = G.subgraph(extended_nodes).copy()

            return subgraph
        except Exception as e:
            print(f"error: {e}")
            return G.subgraph([]).copy()  # 返回空图
    except Exception as e:
        print(f"2hgraph error: {e}")
        return G.subgraph([]).copy() if G else None


# data_processing/umls_utils.py
def batch_get_2hop_subgraph(full_graph, batch_matched_cuis, max_nodes=500):
    try:
        # 输入验证
        if full_graph is None:
            return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]
        if not isinstance(batch_matched_cuis, list):
            return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]
        if not isinstance(max_nodes, int) or max_nodes <= 0:
            max_nodes = 500

        batch_subgraphs = []
        for i, cuis in enumerate(tqdm(batch_matched_cuis, desc="subgraph")):
            try:
                if cuis is None:
                    cuis = []
                elif not isinstance(cuis, (list, set)):
                    print(f"error3")
                    batch_subgraphs.append(full_graph.subgraph([]).copy())
                    continue

                subgraph = get_2hop_subgraph(full_graph, cuis, max_nodes)
                batch_subgraphs.append(subgraph)
            except Exception as e:
                print(f"{e}")
                batch_subgraphs.append(full_graph.subgraph([]).copy())

        return batch_subgraphs
    except Exception as e:
        print(f" {e}")
        return [full_graph.subgraph([]).copy() if full_graph else None for _ in batch_matched_cuis]



