
"""
Script to convert processed PIQA data to PyTorch Dataset and DataLoader format
This script takes the processed_piqa.jsonl file and converts it to a PyTorch Dataset,
then creates training, validation, and test DataLoaders for model training.

Input:
- processed_piqa.jsonl: Contains combined sentences and labels

Output:
- piqa_train_dataset.pt: PyTorch Dataset file
- DataLoaders for training, validation, and testing
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

class CollateFnHelper:
    def __init__(self, max_len=None):  # 保留参数兼容，但不用
        self.max_len = max_len

    def __call__(self, batch):
        sentence_list = [item["input_text"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])  # 堆叠sol1标签\
        original_ids = [item["original_ids"] for item in batch]  # 保留原始ID

        return {
            "input_text": sentence_list,
            "label": labels,
            "original_ids": original_ids
        }

def load_processed_piqa(jsonl_path, sample_size=None):
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
                "sentence": item["sentence"],
                "label": item["label"],
            })
            # 如果达到指定样本数量，停止加载
            if len(processed_data) >= sample_size:
                break
    return processed_data

def save_to_csv_piqa(processed_data, save_path="piqa_fact_data.csv"):
    df = pd.DataFrame(processed_data)[["sentence1", "sentence2", "label1", "label2"]]
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"CSV saved to: {save_path}")
    print(f"Total samples: {len(df)}")
    print(f"Correct sol1 count (label1=1): {df['label1'].sum()}")
    print(f"Correct sol2 count (label2=1): {df['label2'].sum()}")

class PIQADataset(Dataset):
    def __init__(self, processed_data):
        self.sentence1 = [item["sentence"] for item in processed_data]
        self.label1 = [torch.tensor(item["label"], dtype=torch.long) for item in processed_data]
        self.original_ids = [item["id"] for item in processed_data]

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, idx):
        return {
            "input_text": self.sentence1[idx],
            "label": self.label1[idx],
            "original_ids": self.original_ids[idx]
        }

def save_torch_dataset_piqa(processed_data, save_path="piqa_train_dataset.pt"):
    dataset = PIQADataset(processed_data)
    torch.save(dataset, save_path)
    print(f"PyTorch Dataset saved to: {save_path}")
    return dataset

def get_loaders_from_existing_dataset_piqa(
        full_dataset,
        batch_size=8,
        max_len=256,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
):
    """Create DataLoaders with specified train/val/test ratios.
    
    Args:
        full_dataset: Full PyTorch Dataset
        batch_size: Batch size for training loader
        max_len: Maximum sequence length (unused)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # 验证比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    generator = torch.Generator().manual_seed(42)
    total_size = len(full_dataset)
    
    # 计算各数据集的大小
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"\n=== 数据集划分结果 ===")
    print(f"总样本数：{total_size}")
    print(f"训练集：{len(train_dataset)} 样本（{train_ratio * 100:.1f}%）")
    print(f"验证集：{len(val_dataset)} 样本（{val_ratio * 100:.1f}%）")
    print(f"测试集：{len(test_dataset)} 样本（{test_ratio * 100:.1f}%）")

    collate_helper = CollateFnHelper(max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_helper,
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_helper,
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_helper,
        num_workers=0 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 设置文件路径
    processed_jsonl_path = r"c:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\PIQA\physicaliqa-train-dev\processed_piqa.jsonl"
    
    # 控制参数
    sample_size = 500  # 总共只选500个样本
    batch_size = 8
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # 加载处理后的数据（控制样本数量）
    processed_data = load_processed_piqa(processed_jsonl_path, sample_size=sample_size)
    
    print(f"\n📊 数据加载结果")
    print(f"- 原始总样本数：{sum(1 for _ in open(processed_jsonl_path, 'r', encoding='utf-8'))}")
    print(f"- 加载的样本数：{len(processed_data)}")
    
    # 保存为CSV格式
    save_to_csv_piqa(processed_data)
    
    # 转换为PyTorch Dataset
    full_dataset = save_torch_dataset_piqa(processed_data)
    
    # 创建DataLoaders（按8:1:1划分）
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset_piqa(
        full_dataset=full_dataset,
        batch_size=batch_size,
        max_len=256,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # 查看loader返回的数据样子（只看前2个batch，避免输出过多）
    print("=" * 80)
    print("训练集Loader返回数据结构展示（前2个batch）")
    print("=" * 80)
    
    batch_idx = 0
    for batch in train_loader:
        if batch_idx >= 2:  # 只看前2个batch，防止输出刷屏
            break
        
        print(f"\n📦 【第 {batch_idx + 1} 个Batch】")
        print(f"✅ Batch 包含字段：{list(batch.keys())}")  # 查看所有字段名
        print(f"✅ 单个Batch样本数：{len(batch['sentence1'])}（与batch_size一致）")
        
        # 1. 打印句子1
        print(f"\n📝 1. 句子1（共 {len(batch['sentence1'])} 条）：")
        for i, text in enumerate(batch['sentence1']):
            # 文本过长时截断，保留前80字符，末尾加省略号
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"     样本{i + 1}：{display_text}")
        
        # 2. 打印句子2
        print(f"\n📝 2. 句子2（共 {len(batch['sentence2'])} 条）：")
        for i, text in enumerate(batch['sentence2']):
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"     样本{i + 1}：{display_text}")
        
        # 3. 打印标签
        print(f"\n🏷️  3. 标签信息：")
        print(f"     label1形状：{batch['label1'].shape}（[batch_size,]）")
        print(f"     label1数据类型：{batch['label1'].dtype}")
        print(f"     label1值：{batch['label1'].tolist()}")
        print(f"     label2形状：{batch['label2'].shape}（[batch_size,]）")
        print(f"     label2数据类型：{batch['label2'].dtype}")
        print(f"     label2值：{batch['label2'].tolist()}")
        
        # 4. 打印原始ID
        print(f"\n🆔 4. 原始数据ID（共 {len(batch['original_ids'])} 个）：")
        for i, orig_id in enumerate(batch['original_ids']):
            print(f"     样本{i + 1}原始ID：{orig_id}")
        
        # 校验字段长度一致性（避免后续报错）
        assert len(batch['sentence1']) == len(batch['sentence2']) == len(batch['label1']) == len(batch['label2']) == len(batch['original_ids']), \
            "❌ Batch各字段长度不一致！"
        print("✅ 字段长度校验通过：所有字段样本数一致")
        print("-" * 70)
        batch_idx += 1
    
    # 可选：快速查看验证集第一个Batch（简洁版）
    print("\n📊 【验证集第一个Batch简览】")
    try:
        val_batch = next(iter(val_loader))
        print(f"✅ 验证集Batch样本数：{len(val_batch['sentence1'])}")
        print(f"✅ 验证集label1分布：{val_batch['label1'].tolist()}")
        print(f"✅ 验证集label2分布：{val_batch['label2'].tolist()}")
        print(f"✅ 验证集字段：{list(val_batch.keys())}")
    except StopIteration:
        print("⚠️  验证集为空！")
    
    print("=" * 80)
