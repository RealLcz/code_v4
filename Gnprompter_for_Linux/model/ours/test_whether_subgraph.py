from Gnprompter_for_Linux.model.ours.utils import extract_matched_entities, map_keywords_to_cuis, get_2hop_subgraph, \
    batch_get_2hop_subgraph
import pickle
import time

graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"

# 加载图谱并计时
print("加载图谱中...")
start_time = time.time()
with open(graph_path, "rb") as f:
    full_graph = pickle.load(f)
load_time = time.time() - start_time
print(f"图谱加载完成！用时: {load_time:.4f} 秒")

# 准备测试语句
batch_statements = []
statement_of_ASQ_1 = "Human papillomavirus and pterygium. Is the virus a risk factor?"
statement_of_ASQ_2 = "Gender difference in survival of resected non-small cell lung cancer: histology-related phenomenon?"
statement_of_ASQ_3 = "Neoadjuvant Imatinib in Locally Advanced Gastrointestinal stromal Tumours, Will Kit Mutation Analysis Be a Pathfinder?"
statement_of_ASQ_4 = "Can the condition of the cell microenvironment of mediastinal lymph nodes help predict the risk of metastases in non-small cell lung cancer?"
batch_statements.append(statement_of_ASQ_1)
batch_statements.append(statement_of_ASQ_2)
batch_statements.append(statement_of_ASQ_3)
batch_statements.append(statement_of_ASQ_4)

# 准备CUI到名称的映射
cui2name_full = {cui: full_graph.nodes[cui]["name"] for cui in full_graph.nodes()}

# 初始化累计时间
total_extract_time = 0
total_map_time = 0
batch_matched_cuis = []

print("\n开始测试函数性能...")
# 处理每个语句
for i, statement in enumerate(batch_statements):
    print(f"\n处理语句 {i+1}: {statement}")
    
    # 测试 extract_matched_entities 函数（重点测试）
    start_time = time.time()
    entities_of_ASQ = extract_matched_entities(statement, graph_path)
    extract_time = time.time() - start_time
    total_extract_time += extract_time
    print(f"extract_matched_entities 用时: {extract_time:.4f} 秒")
    print(f"提取到的实体数量: {len(entities_of_ASQ)}")
    print(f"提取到的实体前5个: {entities_of_ASQ[:5]}")
    
    # 测试 map_keywords_to_cuis 函数
    start_time = time.time()
    cui_of_statement = map_keywords_to_cuis(entities_of_ASQ, cui2name_full)
    map_time = time.time() - start_time
    total_map_time += map_time
    print(f"map_keywords_to_cuis 用时: {map_time:.4f} 秒")
    print(f"映射到的CUI数量: {len(cui_of_statement)}")
    
    batch_matched_cuis.append(cui_of_statement)

# 测试 batch_get_2hop_subgraph 函数
print("\n测试 batch_get_2hop_subgraph 函数...")
start_time = time.time()
batch_subgraphs = batch_get_2hop_subgraph(full_graph, batch_matched_cuis, 1000)
subgraph_time = time.time() - start_time
print(f"batch_get_2hop_subgraph 用时: {subgraph_time:.4f} 秒")

# 分析子图质量
print("\n子图提取结果分析:")
for i, subgraph in enumerate(batch_subgraphs[:3]):  # 只分析前三个子图
    if subgraph is not None:
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        print(f"语句 {i+1} 的子图: 节点数={num_nodes}, 边数={num_edges}")
    else:
        print(f"语句 {i+1} 的子图为None")

# 汇总性能信息
avg_extract_time = total_extract_time / len(batch_statements)
avg_map_time = total_map_time / len(batch_statements)
avg_subgraph_time = subgraph_time / len(batch_statements)
avg_total_time = avg_extract_time + avg_map_time + avg_subgraph_time

print("\n=== 性能测试汇总 ===")
print(f"总测试语句数: {len(batch_statements)}")
print("\n【重点关注 - extract_matched_entities性能】")
print(f"extract_matched_entities 平均用时: {avg_extract_time:.4f} 秒/语句")
print(f"占总处理时间比例: {avg_extract_time/avg_total_time*100:.2f}%")
print("\n其他函数性能")
print(f"map_keywords_to_cuis 平均用时: {avg_map_time:.4f} 秒/语句")
print(f"batch_get_2hop_subgraph 总用时: {subgraph_time:.4f} 秒 (处理 {len(batch_statements)} 个语句)")
print(f"batch_get_2hop_subgraph 平均用时: {avg_subgraph_time:.4f} 秒/语句")
print(f"整体处理平均用时: {avg_total_time:.4f} 秒/语句")

# 确定性能瓶颈函数
bottleneck_tuple = max([(avg_extract_time, 'extract_matched_entities'), 
                       (avg_map_time, 'map_keywords_to_cuis'), 
                       (avg_subgraph_time, 'batch_get_2hop_subgraph')], 
                     key=lambda x: x[0])
bottleneck = bottleneck_tuple[1]
print(f"\n性能瓶颈函数: {bottleneck} ({bottleneck_tuple[0]:.4f} 秒)")

# 针对extract_matched_entities的优化建议
if bottleneck == "extract_matched_entities":
    print("\n=== extract_matched_entities 优化建议 ===")
    print("1. 考虑使用基于索引的快速版本，避免全量遍历")
    print("2. 对UMLS实体构建倒排索引，加速实体匹配过程")
    print("3. 优化模糊匹配算法，减少计算复杂度")
    print("4. 考虑使用并行处理技术加速实体匹配")

print("\n测试完成！")
