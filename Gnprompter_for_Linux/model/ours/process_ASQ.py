import json
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader



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

def load_and_map_labels(original_json_path):
    processed_data = []
    with open(original_json_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    for item_id, item in original_data.items():
        input_text = item["QUESTION"].strip()
        label = 1 if item["final_decision"].strip() == "yes" else 0
        processed_data.append(
            {
                "original_id": item_id,
                "input_text": input_text,
                "label": label
            }
        )

    return processed_data

def save_to_csv(processed_data, save_path="medical_fact_data.csv"):
    df = pd.DataFrame(processed_data)[["input_text", "label"]]
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"CSV save to:{save_path}")
    print(f"训练样本数：{len(df)}，正样本数（label=1）：{df['label'].sum()}，负样本数（label=0）：{len(df) - df['label'].sum()}")

def save_to_jsonl(processed_data, save_path="medical_fact_data.jsonl"):
    with open(save_path, "w", encoding="utf-8") as f:
        for data in processed_data:
            train_item = {"input_text": data["input_text"], "label": data["label"]}
            json.dump(train_item, f, ensure_ascii=False)
            f.write("\n")
    print(f"JSONL文件已保存：{save_path}")

class MedicalFactDataset(Dataset):
    def __init__(self, processed_data):
        self.input_texts = [item["input_text"] for item in processed_data]
        self.labels = [torch.tensor(item["label"], dtype=torch.long) for item in processed_data]
        self.original_ids = [item["original_id"] for item in processed_data]

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return {
            "input_text": self.input_texts[idx],
            "label": self.labels[idx],
            "original_ids": self.original_ids[idx]
        }

def save_torch_dataset(processed_data, save_path="medical_train_dataset.pt"):
    dataset = MedicalFactDataset(processed_data)
    torch.save(dataset, save_path)
    print(f"PyTorch Dataset已保存：{save_path}")
    return dataset

def get_loaders_from_existing_dataset(
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


if __name__ == '__main__':
    # 这里放之前的执行代码（从创建loader开始）
    original_json_path = r"C:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\ASQ\ori_pqal.json"
    processed_data = load_and_map_labels(original_json_path)
    save_to_csv(processed_data)
    save_to_jsonl(processed_data)
    full_dataset = save_torch_dataset(processed_data)

    # 创建Loader：关键修改 num_workers=0（Windows下禁用多进程）
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset(
        full_dataset=full_dataset,
        batch_size=8,
        max_len=128
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
        print(f"✅ 单个Batch样本数：{len(batch['input_text'])}（与batch_size一致）")

        # 1. 打印输入文本（input_text）
        print(f"\n📝 1. 输入文本（共 {len(batch['input_text'])} 条）：")
        for i, text in enumerate(batch['input_text']):
            # 文本过长时截断，保留前80字符，末尾加省略号
            display_text = text[:80] + "..." if len(text) > 80 else text
            print(f"     样本{i + 1}：{display_text}")

        # 2. 打印标签（label）
        print(f"\n🏷️  2. 标签信息：")
        print(f"     标签形状：{batch['label'].shape}（[batch_size,]）")
        print(f"     标签数据类型：{batch['label'].dtype}")
        print(f"     标签值（0=负样本/1=正样本）：{batch['label'].tolist()}")

        # 3. 打印原始ID（original_ids）
        print(f"\n🆔 3. 原始数据ID（共 {len(batch['original_ids'])} 个）：")
        for i, orig_id in enumerate(batch['original_ids']):
            print(f"     样本{i + 1}原始ID：{orig_id}")

        # 校验字段长度一致性（避免后续报错）
        assert len(batch['input_text']) == len(batch['label']) == len(batch['original_ids']), \
            "❌ Batch各字段长度不一致！"
        print("✅ 字段长度校验通过：所有字段样本数一致")
        print("-" * 70)
        batch_idx += 1

    # 可选：快速查看验证集第一个Batch（简洁版）
    print("\n📊 【验证集第一个Batch简览】")
    try:
        val_batch = next(iter(val_loader))
        print(f"✅ 验证集Batch样本数：{len(val_batch['input_text'])}")
        print(f"✅ 验证集标签分布：{val_batch['label'].tolist()}")
        print(f"✅ 验证集字段：{list(val_batch.keys())}")
    except StopIteration:
        print("⚠️  验证集为空！")

    print("=" * 80)