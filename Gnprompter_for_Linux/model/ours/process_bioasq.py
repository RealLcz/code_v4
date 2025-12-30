import json
import re

import torch
from torch.utils.data import Dataset, DataLoader, random_split

def process_bioasq(input_path, output_path):
    """
    处理BioASQ数据集，生成正样本和负样本
    正样本：body + ideal_answer，label=1
    负样本：body + not + ideal_answer，label=0
    """
    try:
        # 打开输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            # 读取整个JSON
            data = json.load(f)
            
        # 获取questions数组
        questions = data.get("questions", [])
        print(f"找到 {len(questions)} 个问题")
        
        processed_count = 0
        
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for question in questions:
                index_bioasq = 0
                body = question.get("body", "")
                ideal_answers = question.get("ideal_answer", [])
                
                # 只处理有body和ideal_answer的问题
                if body and ideal_answers:
                    # 取第一个ideal_answer
                    ideal_answer = ideal_answers[0].strip()
                    
                    # 生成正样本
                    positive_text = f"{body} {ideal_answer}"
                    positive_sample = {
                        "id": index_bioasq,
                        "combined_text": positive_text,
                        "label": 1
                    }
                    f_out.write(json.dumps(positive_sample, ensure_ascii=False) + '\n')
                    
                    # 生成负样本
                    negative_text = f"{body} not {ideal_answer}"
                    negative_sample = {
                        "id": index_bioasq + 1,
                        "combined_text": negative_text,
                        "label": 0
                    }
                    index_bioasq = index_bioasq + 2
                    f_out.write(json.dumps(negative_sample, ensure_ascii=False) + '\n')
                    
                    processed_count += 1
                    
                    if processed_count % 1000 == 0:
                        print(f"已处理 {processed_count} 个问题")
        
        print(f"处理完成！共生成 {processed_count * 2} 个样本")
        print(f"输出文件：{output_path}")
        
    except Exception as e:
        print(f"处理过程中发生错误：{e}")

def load_processed_bioasq(jsonl_path, sample_size=None):
    """Load processed PIQA data from JSONL file.

    Args:
        jsonl_path: Path to the processed PIQA JSONL file
        sample_size: Number of samples to load (None means load all)

    Returns:
        List of processed PIQA data
    """
    processed_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            processed_data.append({
                "id": item["id"],
                "sentence": item["combined_text"],
                "label": item["label"],
            })
            # 如果达到指定样本数量，停止加载
            if len(processed_data) >= sample_size:
                break
    return processed_data

class BioasqDataset(Dataset):
    def __init__(self, processed_data):
        self.input_texts = [item["sentence"] for item in processed_data]
        self.labels = [torch.tensor(item["label"], dtype=torch.long) for item in processed_data]
        self.original_ids = [item["id"] for item in processed_data]

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return {
            "input_text": self.input_texts[idx],
            "label": self.labels[idx],
            "original_ids": self.original_ids[idx]
        }

def save_torch_dataset_bioasq(processed_data, save_path="piqa_train_dataset.pt"):
    dataset = BioasqDataset(processed_data)
    torch.save(dataset, save_path)
    print(f"PyTorch Dataset saved to: {save_path}")
    return dataset

class CollateFnHelper:
    def __init__(self, max_len=None):  # 保留参数兼容，但不用
        self.max_len = max_len

    def __call__(self, batch):
        # 只批量整理原始字段，不做编码（编码移到模型内部）
        input_texts = [item["input_text"] for item in batch]  # 保留原始文本
        labels = torch.stack([item["label"] for item in batch])  # 堆叠标签
        original_ids = [item["original_ids"] for item in batch]  # 保留原始ID

        return {
            "input_text": input_texts,  # 关键：保留input_text字段
            "label": labels,
            "original_ids": original_ids
        }

def get_loaders_from_existing_dataset_bioasq(
        full_dataset,
        batch_size=8,
        max_len=256,
        test_size=0.1,
        val_size=0.1
):
    generator = torch.Generator().manual_seed(42)
    total_size = len(full_dataset)
    train_size = int(total_size * (1 - test_size - val_size))
    val_size = int(total_size * val_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"\n=== 数据集划分结果 ===")
    print(f"训练集：{len(train_dataset)} 样本（80%）")
    print(f"验证集：{len(val_dataset)} 样本（10%）")
    print(f"测试集：{len(test_dataset)} 样本（10%）")

    collate_helper = CollateFnHelper(max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_helper,  # 直接用实例
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_helper,  # 直接用实例
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_helper,  # 直接用实例
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader