import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 输入和输出文件路径
input_file = r"/root/autodl-tmp/Gnprompter_for_Linux/dataset/Riddle/rs_train.jsonl"
output_file = r"/root/autodl-tmp/Gnprompter_for_Linux/dataset/Riddle/rs_train.jsonl"

try:
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        line_count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON行
                data = json.loads(line)
                
                # 获取问题stem
                stem = data["question"]["stem"]
                
                # 获取正确答案标签
                correct_label = data["answerKey"]
                
                # 处理每个选项
                for choice in data["question"]["choices"]:
                    label = choice["label"]
                    text = choice["text"]
                    
                    # 拼接stem和text
                    combined_text = f"{stem} {text}"
                    
                    # 设置标签：正确答案为1，否则为0
                    is_correct = 1 if label == correct_label else 0
                    
                    # 构建新的JSON对象
                    processed_data = {
                        "id": data["id"],
                        "combined_text": combined_text,
                        "is_correct": is_correct
                    }
                    
                    # 写入输出文件
                    f_out.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"已处理 {line_count} 行")
                    
            except json.JSONDecodeError as e:
                print(f"解析JSON错误：{e}，行内容：{line}")
            except KeyError as e:
                print(f"缺少键：{e}，行内容：{line}")
    
    print(f"处理完成，共处理 {line_count} 行数据")
    print(f"输出文件：{output_file}")
    
except FileNotFoundError:
    print(f"文件未找到：{input_file}")
except Exception as e:
    print(f"处理过程中发生错误：{e}")


def load_processed_riddle(jsonl_path, sample_size=None):
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
                "label": item["is_correct"],
            })
            # 如果达到指定样本数量，停止加载
            if len(processed_data) >= sample_size:
                break
    return processed_data


class RiddleDataset(Dataset):
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

def save_torch_dataset_riddle(processed_data, save_path="piqa_train_dataset.pt"):
    dataset = RiddleDataset(processed_data)
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

def get_loaders_from_existing_dataset_riddle(
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
