import argparse
import pickle

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim, nn
from tqdm import tqdm

from utils import fast_extract_matched_entities
from utils import map_keywords_to_cuis_new,\
    batch_get_2hop_subgraph
from data_processing.change_umls_into_needed import generate_pos_neg_triples, \
    batch_umls_graph_to_pyg_data
from MyModel import MyModel
from data_processing.process_ASQ import load_and_map_labels, save_to_csv, save_to_jsonl, save_torch_dataset, \
    get_loaders_from_existing_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


parser = argparse.ArgumentParser()
# parameters of GAT
parser.add_argument("--use_relational_gnn", type=bool, default=True)
parser.add_argument("--gnn_edge_dim",type=int, default=1024)
parser.add_argument("--gnn_in_dim", type=int, default=4096)    # equal to Qwen embeder dim
parser.add_argument("--gnn_hidden_dim", type=int, default=4096)
parser.add_argument("--gnn_out_dim", type=int, default=4096)
parser.add_argument("--gnn_heads", type=int, default=8)
parser.add_argument("--gnn_layers", type=int, default=3)
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument("--upcast", type=bool, default=False)
parser.add_argument("--n_ntype", type=int, default=127)
parser.add_argument("--n_etype", type=int, default=11)

# parameters of LLM
parser.add_argument("--llm_model_path", type=str, default="Qwen/Qwen3-4B")
parser.add_argument("--llm_emb_dim", type=int, default=2560) # equal to Qwen's hidden layer dim
parser.add_argument("--llm_vocab_size", type=int, default=151936) # equal to the size of Qwen's dictionary
parser.add_argument("--cross_attention_layers", type=int, default=8)
parser.add_argument("--dim_input", type=int ,default=2560)

# parameters of Link Prediction
parser.add_argument("--link_gamma", type=float, default=12.0)
parser.add_argument("--negative_adversarial_sampling", type=bool, default=True)
parser.add_argument("--adversarial_temperature", type=float, default=1.0)
parser.add_argument("--reg_param", type=float, default=0.01)

# path parameter
parser.add_argument("--graph_path", type=str, default=r"/root/autodl-tmp/Gnprompter_for_Linux/dataset/umls_old/processed_umls_big1500000.pkl")
parser.add_argument("--index_path", type=str, default=r"/root/autodl-tmp/Gnprompter_for_Linux/model/ours/umls_entity_index_1500000.pkl")

# training parameters
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--link_loss_weight", type=float, default=0.3)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=64)

# 在训练脚本的parser添加以下参数（和其他参数并列）
parser.add_argument("--moe_input_size", type=int, default=2560)  # 和dim_input一致（Qwen嵌入维度）
parser.add_argument("--moe_hidden_size", type=int, default=2048)  # 专家网络隐藏层维度（可调整）
parser.add_argument("--moe_output_size", type=int, default=2560)  # 和输入维度一致（保持维度不变）
parser.add_argument("--moe_num_experts", type=int, default=8)  # 专家数量（根据GPU显存调整，8/16/32）
parser.add_argument("--moe_k", type=int, default=4)  # 每个样本选择的专家数（≤num_experts）
parser.add_argument("--moe_loss_weight", type=float, default=0.1)  # MoE负载均衡损失权重
parser.add_argument("--save_model_path", type=str, default="./checkpoints/my_model_best")  # 补充保存路径参数（之前缺失）
parser.add_argument("--max_text_len", type=int, default=32)  # 补充缺失的文本长度参数
parser.add_argument("--neg_sample_size", type=int, default=10)  # 补充负样本大小参数
parser.add_argument("--max_question_len", type=int, default=32)


args = parser.parse_args()

original_json_path = r"/root/autodl-tmp/Gnprompter_for_Linux/dataset/PQA/pqal_fold1/train_set.json"
processed_data = load_and_map_labels(original_json_path)
save_to_csv(processed_data)
save_to_jsonl(processed_data)
full_dataset = save_torch_dataset(processed_data)

# 初始化Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
LLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").to(device)
for param in LLM.parameters():
    param.requires_grad = False

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 创建Loader
train_loader, val_loader, test_loader = get_loaders_from_existing_dataset(
    full_dataset=full_dataset,
    batch_size=args.batch_size,
    max_len=128
)

def train(args, model, train_loader, full_graph, cui2name_full, optimizer, criterion, name_to_cuis, text_projector):
    model.train()
    total_loss = 0.0
    train_preds, train_labels = [], []

    for batch in tqdm(train_loader, desc="Training"):
        question_text = batch["input_text"]  # [B, ]
        labels = batch["label"].to(args.device)  # [B, ]
        B = len(question_text)

        # --------------------------
        # 1. 批量提取关键词
        # --------------------------
        batch_keywords = []
        for q in question_text:
            keywords = fast_extract_matched_entities(q, args.index_path)
            batch_keywords.append(list(set(keywords)))  # 去重后存入批量列表

        # --------------------------
        # 2. 批量映射CUI
        # --------------------------
        batch_matched_cuis = map_keywords_to_cuis_new(batch_keywords, name_to_cuis)

        # 检查是否所有问题都没有提取到实体
        has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
        
        if not has_entities:
            # 所有问题都没有提取到实体，生成全0向量
            print(f"警告：当前批次没有提取到任何实体，使用全0向量作为输入")
            # 生成形状为[B, N, llm_emb_dim]的全0向量，N取默认值50
            N = 50
            llm_emb_dim = args.llm_emb_dim
            multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
            link_loss = torch.tensor(0.0).to(args.device)  # 链路损失设为0
        else:
            # --------------------------
            # 3. 批量提取2跳子图（核心修改）
            # --------------------------
            batch_subgraphs = batch_get_2hop_subgraph(
                full_graph=full_graph,
                batch_matched_cuis=batch_matched_cuis,
                max_nodes=50
            )  # [B]，每个元素是子图（NetworkX对象）

            print(f"2hop subgraph 提取完成，开始转换成PyG数据...")

            # --------------------------
            # 4. 批量转换为PyG数据（复用已有函数）
            # --------------------------


            gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                batch_subgraphs=batch_subgraphs,
                args=args,
                device=args.device,
                tokenizer=tokenizer,
                model=LLM,
                project_layer=text_projector
            )

            print(f"转换为PyG数据结束，进入grasper模块...")

            # --------------------------
            # 5. 批量生成多模态嵌入和链路损失
            # --------------------------
            multimodal_emb, link_loss = model.graph_grasper(
                gat_inputs=gat_inputs,
                graphs=batch_pyg_graphs,  # 传入批量PyG图
                text_attributes=batch_text_attrs,
                triples_list=batch_triples  # 传入批量三元组
            )

        # --------------------------
        # 6. 后续流程保持不变（批量处理）
        # --------------------------
        question_embedding = model.question_embeder(question_text)
        the_prompt, moe_loss = model.cross_modality_MHA(multimodal_emb, question_embedding)
        # the prompt [B, N, d]
        # question_embedding [B, seq_len_q, d]
        input_embeds = torch.cat([question_embedding, the_prompt], dim=1)
        outputs = model.model(inputs_embeds=input_embeds, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]
        cls_hidden = hidden_states[:, -1, :]
        logits = model.classifier(cls_hidden)
        cls_loss = criterion(logits, labels)

        # 总损失（链路损失已在graph_grasper中批量计算）
        total_batch_loss = cls_loss + (args.link_loss_weight * link_loss if link_loss is not None else 0) + (moe_loss if moe_loss is not None else 0)

        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # 记录结果
        total_loss += total_batch_loss.item() * B
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

    # 计算指标（保持不变）
    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(train_labels, train_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1

def validate(args, tokenizer, LLM, model, val_loader, full_graph, cui2name_full, criterion, name_to_cuis, text_projector):
    model.eval()
    total_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            question_text = batch["input_text"]  # [B, ]
            labels = batch["label"].to(args.device)  # [B, ]
            B = len(question_text)

            # 1. 批量提取关键词
            batch_keywords = [list(set(fast_extract_matched_entities(q, args.graph_path))) for q in question_text]

            # 2. 批量映射CUI
            batch_matched_cuis = [map_keywords_to_cuis_new(kw, name_to_cuis) for kw in batch_keywords]
            
            # 检查是否所有问题都没有提取到实体
            has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
            
            if not has_entities:
                # 所有问题都没有提取到实体，生成全0向量
                print(f"警告：当前批次没有提取到任何实体，使用全0向量作为输入")
                # 生成形状为[B, N, llm_emb_dim]的全0向量，N取默认值50
                N = 50
                llm_emb_dim = args.llm_emb_dim
                multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
                link_loss = torch.tensor(0.0).to(args.device)  # 链路损失设为0
            else:
                # 3. 批量提取子图
                batch_subgraphs = batch_get_2hop_subgraph(
                    full_graph=full_graph,
                    batch_matched_cuis=batch_matched_cuis,
                    max_nodes=500
                )

                # 4. 批量转换为PyG数据
                gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                    batch_subgraphs=batch_subgraphs,
                    args=args,
                    device=args.device,
                    tokenizer=tokenizer,
                    model=LLM,
                    project_layer=text_projector
                )

                # 5. 批量生成多模态嵌入和链路损失
                multimodal_emb, link_loss = model.graph_grasper(
                    gat_inputs=gat_inputs,
                    graphs=batch_pyg_graphs,
                    text_attributes=batch_text_attrs,
                    triples_list=batch_triples
                )

            # 后续流程（与train相同）
            question_embedding = model.question_embeder(question_text)
            the_prompt, moe_loss = model.cross_modality_MHA(multimodal_emb, question_embedding)
            input_embeds = torch.cat([question_embedding.unsqueeze(1), the_prompt], dim=1)
            outputs = model.model(input_embeds=input_embeds, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            cls_hidden = hidden_states[:, -1, :]
            logits = model.classifier(cls_hidden)
            cls_loss = criterion(logits, labels)
            total_batch_loss = cls_loss + args.link_loss_weight * link_loss + moe_loss  # 验证时可保留moe_loss

            # 记录结果
            total_loss += total_batch_loss.item() * B
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    # 计算指标（保持不变）
    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(val_labels, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1


def test(args, model, test_loader, full_graph, criterion, name_to_cuis, text_projector):
    model.eval()
    total_loss = 0.0
    test_preds, test_labels = [], []
    test_original_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            question_text = batch["input_text"]  # [B, ]
            labels = batch["label"].to(args.device)  # [B, ]
            original_ids = batch["original_id"]  # 原始ID
            B = len(question_text)

            # 1. 批量提取关键词
            batch_keywords = [list(set(fast_extract_matched_entities(q, r"/root/autodl-tmp/Gnprompter_for_Linux/dataset/umls_old/processed_umls_big1500000.pkl"))) for q in question_text]

            # 2. 批量映射CUI
            batch_matched_cuis = [map_keywords_to_cuis_new(kw, name_to_cuis) for kw in batch_keywords]
            
            # 检查是否所有问题都没有提取到实体
            has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
            
            if not has_entities:
                # 所有问题都没有提取到实体，生成全0向量
                print(f"警告：当前批次没有提取到任何实体，使用全0向量作为输入")
                # 生成形状为[B, N, llm_emb_dim]的全0向量，N取默认值50
                N = 50
                llm_emb_dim = args.llm_emb_dim
                multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
                link_loss = torch.tensor(0.0).to(args.device)  # 链路损失设为0
            else:
                # 3. 批量提取子图
                batch_subgraphs = batch_get_2hop_subgraph(
                    full_graph=full_graph,
                    batch_matched_cuis=batch_matched_cuis,
                    max_nodes=500
                )

                # 4. 批量转换为PyG数据
                gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                    batch_subgraphs=batch_subgraphs,
                    args=args,
                    device=args.device,
                    tokenizer=tokenizer,
                    model=LLM,
                    project_layer=text_projector
                )

                # 5. 批量生成多模态嵌入和链路损失
                multimodal_emb, link_loss = model.graph_grasper(
                    gat_inputs=gat_inputs,
                    graphs=batch_pyg_graphs,
                    text_attributes=batch_text_attrs,
                    triples_list=batch_triples
                )

            # 后续流程（与train相同）
            question_embedding = model.question_embeder(question_text)
            the_prompt, moe_loss = model.cross_modality_MHA(multimodal_emb, question_embedding)
            input_embeds = torch.cat([question_embedding.unsqueeze(1), the_prompt], dim=1)
            outputs = model.model(input_embeds=input_embeds, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            cls_hidden = hidden_states[:, -1, :]
            logits = model.classifier(cls_hidden)
            cls_loss = criterion(logits, labels)
            total_batch_loss = cls_loss + args.link_loss_weight * link_loss + moe_loss

            # 记录结果
            total_loss += total_batch_loss.item() * B
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
            test_original_ids.extend(original_ids)

    # 计算指标+保存结果（保持不变）
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0
    )

    # 保存测试结果
    import pandas as pd
    test_results = pd.DataFrame({
        "original_id": test_original_ids,
        "question_text": [q for batch in test_loader for q in batch["input_text"]],
        "true_label": test_labels,
        "pred_label": test_preds,
        "pred_result": ["yes" if p == 1 else "no" for p in test_preds]
    })
    test_results.to_csv("test_results.csv", index=False, encoding="utf-8-sig")
    print(f"测试结果已保存到：test_results.csv")

    return avg_loss, acc, prec, rec, f1


def main(args):
    print(f"训练配置：{args}")

    # --------------------------
    # 关键修改：加载full_graph和cui2name_full
    # --------------------------
    # 加载完整图谱（NetworkX对象）
    try:
        with open(args.graph_path, "rb") as f:
            full_graph = pickle.load(f)
        # 构建CUI到名称的映射（用于关键词匹配）
        cui2name_full = {cui: full_graph.nodes[cui]["name"] for cui in full_graph.nodes()}
    except FileNotFoundError:
        print(f"错误：找不到图谱文件 {args.graph_path}，请检查路径是否正确")
        return

    # 建立反向映射：节点名称（小写）→CUI列表（支持同名节点）
    name_to_cuis = {}
    for cui, node_name in cui2name_full.items():
        if node_name:  # 过滤空名称的节点
            name_lower = node_name.lower()
            if name_lower not in name_to_cuis:
                name_to_cuis[name_lower] = []
            name_to_cuis[name_lower].append(cui)

    with open("global_mapping.pkl", "rb") as f:
        global_mapping = pickle.load(f)
    global_n_ntype = len(global_mapping["sem_type2global_idx"])
    global_n_etype = len(global_mapping["rel_id2global_idx"])

    args.n_ntype = global_n_ntype
    args.n_etype = global_n_etype + 1

    print(args.n_ntype)
    print(args.n_etype)

    # 初始化模型
    text_project_layer = nn.Linear(2560, args.gnn_in_dim).to(device)
    model = MyModel(args, tokenizer, LLM).to(args.device)  # 注意传入tokenizer和model

    # 选择可训练参数
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch{epoch + 1}/{args.epochs} ===")

        # 训练：传入full_graph和cui2name_full
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            args, model, train_loader, full_graph, cui2name_full, optimizer, criterion, name_to_cuis, text_project_layer
        )
        print(f"训练集：Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")

        # 验证：传入full_graph和cui2name_full
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            args, tokenizer, LLM, model, val_loader, full_graph, cui2name_full, criterion, name_to_cuis, text_project_layer
        )
        print(f"验证集：Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # 保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1
            }, args.save_model_path + ".pth")
            print(f"保存最优模型（验证集F1={best_val_f1:.4f}）")

    # 测试最优模型
    print(f"\n=== 测试最优模型 ===")
    checkpoint = torch.load(args.save_model_path + ".pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载epoch {checkpoint['epoch']} 的模型（验证集F1={checkpoint['best_val_f1']:.4f}）")

    test_loss, test_acc, test_prec, test_rec, test_f1 = test(
        args, model, test_loader, full_graph, criterion, name_to_cuis, text_project_layer
    )

    print(f"\n测试集最终性能：")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

    print(f"\n训练-验证-测试流程完成！最优模型保存路径：{args.save_model_path}.pth")


if __name__ == "__main__":
    main(args)
