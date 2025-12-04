import pickle

from tqdm import tqdm

from Gnprompter_for_Linux.model.ours.utils import fast_extract_matched_entities, map_keywords_to_cuis_new
from data_processing.process_ASQ import load_and_map_labels, save_to_csv, save_to_jsonl, \
    save_torch_dataset, get_loaders_from_existing_dataset

def count_missing_keywords_and_cuis(loader, index_path, name_to_cuis):
    """
    统计数据集中找不到关键词或无法映射到CUI的样本数量
    """
    total_samples = 0
    no_keywords_count = 0  # 找不到关键词的样本数
    keywords_no_cuis_count = 0  # 有关键词但无法映射到CUI的样本数
    
    # 存储无关键词或无CUI映射的样本
    no_keywords_samples = []
    keywords_no_cuis_samples = []
    
    for batch in tqdm(loader, desc="统计关键词缺失情况"):
        question_text = batch["input_text"]  # [B, ]
        labels = batch["label"]  # [B, ]
        total_samples += len(question_text)
        
        for i, q in enumerate(question_text):
            # 1. 提取关键词
            keywords = list(set(fast_extract_matched_entities(q, index_path)))
            
            # 检查是否没有关键词
            if len(keywords) == 0:
                no_keywords_count += 1
                no_keywords_samples.append({
                    'question': q,
                    'label': labels[i]
                })
                continue
            
            # 2. 映射CUI
            matched_cuis = map_keywords_to_cuis_new([keywords], name_to_cuis)[0]
            
            # 检查是否无法映射到CUI
            if len(matched_cuis) == 0:
                keywords_no_cuis_count += 1
                keywords_no_cuis_samples.append({
                    'question': q,
                    'keywords': keywords,
                    'label': labels[i]
                })
    
    # 计算百分比
    no_keywords_percent = (no_keywords_count / total_samples) * 100 if total_samples > 0 else 0
    keywords_no_cuis_percent = (keywords_no_cuis_count / total_samples) * 100 if total_samples > 0 else 0
    total_missing_percent = ((no_keywords_count + keywords_no_cuis_count) / total_samples) * 100 if total_samples > 0 else 0
    
    # 打印统计结果
    print("===== 关键词和CUI映射统计结果 =====")
    print(f"总样本数: {total_samples}")
    print(f"无关键词样本数: {no_keywords_count} ({no_keywords_percent:.2f}%)")
    print(f"有关键词但无CUI映射样本数: {keywords_no_cuis_count} ({keywords_no_cuis_percent:.2f}%)")
    print(f"总计缺失样本数: {no_keywords_count + keywords_no_cuis_count} ({total_missing_percent:.2f}%)")
    
    # 显示一些示例
    print("\n===== 无关键词示例 =====")
    for i, sample in enumerate(no_keywords_samples[:3]):  # 显示前3个示例
        print(f"示例 {i+1}: {sample['question']} (标签: {sample['label']})")
    
    print("\n===== 有关键词但无CUI映射示例 =====")
    for i, sample in enumerate(keywords_no_cuis_samples[:3]):  # 显示前3个示例
        print(f"示例 {i+1}: {sample['question']}")
        print(f"  关键词: {sample['keywords']}")
        print(f"  标签: {sample['label']}")
    
    # 返回统计数据，方便进一步处理
    return {
        'total_samples': total_samples,
        'no_keywords_count': no_keywords_count,
        'keywords_no_cuis_count': keywords_no_cuis_count,
        'no_keywords_samples': no_keywords_samples,
        'keywords_no_cuis_samples': keywords_no_cuis_samples
    }

if __name__ == "__main__":
    # 配置路径
    graph_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\umls\processed_umls_big1500000.pkl"
    original_json_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\PQA\pqal_fold1\train_set.json"
    index_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\model\ours\umls_entity_index_1500000.pkl"
    
    # 加载和处理数据
    processed_data = load_and_map_labels(original_json_path)
    full_dataset = save_torch_dataset(processed_data)
    
    # 创建Loader
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset(
        full_dataset=full_dataset,
        batch_size=64,
        max_len=128
    )
    
    # 加载图谱并构建名称到CUI的映射
    with open(graph_path, "rb") as f:
        full_graph = pickle.load(f)
    
    # 构建CUI到名称的映射（用于关键词匹配）
    cui2name_full = {cui: full_graph.nodes[cui]["name"] for cui in full_graph.nodes()}
    
    # 构建名称到CUI的映射
    name_to_cuis = {}
    for cui, node_name in cui2name_full.items():
        if node_name:  # 过滤空名称的节点
            name_lower = node_name.lower()
            if name_lower not in name_to_cuis:
                name_to_cuis[name_lower] = []
            name_to_cuis[name_lower].append(cui)
    
    # 统计训练集
    print("\n统计训练集关键词缺失情况:")
    train_stats = count_missing_keywords_and_cuis(train_loader, index_path, name_to_cuis)
    
    # 统计验证集
    print("\n统计验证集关键词缺失情况:")
    val_stats = count_missing_keywords_and_cuis(val_loader, index_path, name_to_cuis)
    
    # 统计测试集
    print("\n统计测试集关键词缺失情况:")
    test_stats = count_missing_keywords_and_cuis(test_loader, index_path, name_to_cuis)
    
    # 合并统计结果
    overall_stats = {
        'total_samples': train_stats['total_samples'] + val_stats['total_samples'] + test_stats['total_samples'],
        'no_keywords_count': train_stats['no_keywords_count'] + val_stats['no_keywords_count'] + test_stats['no_keywords_count'],
        'keywords_no_cuis_count': train_stats['keywords_no_cuis_count'] + val_stats['keywords_no_cuis_count'] + test_stats['keywords_no_cuis_count']
    }
    
    # 计算整体百分比
    overall_no_keywords_percent = (overall_stats['no_keywords_count'] / overall_stats['total_samples']) * 100 if overall_stats['total_samples'] > 0 else 0
    overall_keywords_no_cuis_percent = (overall_stats['keywords_no_cuis_count'] / overall_stats['total_samples']) * 100 if overall_stats['total_samples'] > 0 else 0
    overall_missing_percent = ((overall_stats['no_keywords_count'] + overall_stats['keywords_no_cuis_count']) / overall_stats['total_samples']) * 100 if overall_stats['total_samples'] > 0 else 0
    
    # 打印整体统计结果
    print("\n===== 整体关键词和CUI映射统计结果 =====")
    print(f"总样本数: {overall_stats['total_samples']}")
    print(f"无关键词样本数: {overall_stats['no_keywords_count']} ({overall_no_keywords_percent:.2f}%)")
    print(f"有关键词但无CUI映射样本数: {overall_stats['keywords_no_cuis_count']} ({overall_keywords_no_cuis_percent:.2f}%)")
    print(f"总计缺失样本数: {overall_stats['no_keywords_count'] + overall_stats['keywords_no_cuis_count']} ({overall_missing_percent:.2f}%)")
    
    # 保存统计结果到文件
    with open('keyword_missing_stats.pkl', 'wb') as f:
        pickle.dump({
            'train': train_stats,
            'val': val_stats,
            'test': test_stats,
            'overall': overall_stats
        }, f)
    
    print("\n统计结果已保存到 'keyword_missing_stats.pkl'")