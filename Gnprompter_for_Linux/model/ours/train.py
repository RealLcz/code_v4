import argparse
import pickle
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim, nn
from tqdm import tqdm

from data_processing.process_obqa import process_obqa, load_processed_obqa, \
    save_torch_dataset_OBQA, get_loaders_from_existing_dataset_obqa
from data_processing.process_bioasq import load_processed_bioasq, \
    save_torch_dataset_bioasq, get_loaders_from_existing_dataset_bioasq, process_bioasq
from data_processing.process_riddle import load_processed_riddle, \
    save_torch_dataset_riddle, get_loaders_from_existing_dataset_riddle
from data_processing.convert_piqa_to_loader import save_torch_dataset_piqa, \
    get_loaders_from_existing_dataset_piqa
from data_processing.convert_piqa_to_loader import load_processed_piqa
from utils import fast_extract_matched_entities
from utils import map_keywords_to_cuis_new,\
    batch_get_2hop_subgraph
from data_processing.change_umls_into_needed import batch_umls_graph_to_pyg_data
from MyModel import MyModel
from data_processing.process_ASQ import load_and_map_labels, save_to_csv, save_to_jsonl, save_torch_dataset, \
    get_loaders_from_existing_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


parser = argparse.ArgumentParser()
# parameters of GAT
parser.add_argument("--use_relational_gnn", type=bool, default=True)
parser.add_argument("--gnn_edge_dim",type=int, default=1024)
parser.add_argument("--gnn_in_dim", type=int, default=512)     #4096???
parser.add_argument("--gnn_hidden_dim", type=int, default=512) #4096???
parser.add_argument("--gnn_out_dim", type=int, default=512)    #4096???
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
parser.add_argument("--cross_attention_layers", type=int, default=1)
parser.add_argument("--dim_input", type=int ,default=2560)

# parameters of Link Prediction
parser.add_argument("--link_gamma", type=float, default=12.0)
parser.add_argument("--negative_adversarial_sampling", type=bool, default=True)
parser.add_argument("--adversarial_temperature", type=float, default=1.0)
parser.add_argument("--reg_param", type=float, default=0.01)

# path parameter
parser.add_argument("--graph_path", type=str, default=None)
parser.add_argument("--index_path", type=str, default=None)

# training parameters
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--link_loss_weight", type=float, default=0.15)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=18)

parser.add_argument("--moe_input_size", type=int, default=2560)
parser.add_argument("--moe_hidden_size", type=int, default=2048)  
parser.add_argument("--moe_output_size", type=int, default=2560)  
parser.add_argument("--moe_num_experts", type=int, default=8)  
parser.add_argument("--moe_k", type=int, default=4)  
parser.add_argument("--moe_loss_weight", type=float, default=0.1)  
parser.add_argument("--save_model_path", type=str, default="./checkpoints/my_model_best")
parser.add_argument("--max_text_len", type=int, default=32) 
parser.add_argument("--neg_sample_size", type=int, default=10) 
parser.add_argument("--max_question_len", type=int, default=32)
parser.add_argument("--kgs", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--sample_size", type=int, default=2500)

args = parser.parse_args()


if args.kgs == "umls":
    args.graph_path = r"/dataset/umls_old/processed_umls_big1500000.pkl"
    args.index_path = r"umls_entity_index_1500000.pkl"

if args.kgs == "NELL":
    args.graph_path = r"/dataset/NELL/processed_NELL"
    args.index_path = r"nell_entity_index.pkl"


if args.dataset == "PQA":
    original_json_path = r"/dataset/PQA/pqal_fold1/train_set.json"
    processed_data = load_and_map_labels(original_json_path)
    save_to_csv(processed_data)
    save_to_jsonl(processed_data)
    full_dataset = save_torch_dataset(processed_data)
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset(
        full_dataset=full_dataset,
        batch_size=args.batch_size,
        max_len=128
    )

if args.dataset == "PIQA":
    original_json_path = r"/dataset/PIQA/processed_piqa.jsonl"
    processed_data = load_processed_piqa(original_json_path, sample_size=args.sample_size)
    full_dataset = save_torch_dataset_piqa(processed_data)
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset_piqa(
        full_dataset=full_dataset,
        batch_size=args.batch_size,
        max_len=128
    )

if args.dataset == "Riddle":
    original_json_path = r"/dataset/Riddle/processed_Riddle.jsonl"
    processed_data = load_processed_riddle(original_json_path, sample_size=args.sample_size)
    full_dataset = save_torch_dataset_riddle(processed_data)
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset_riddle(
        full_dataset=full_dataset,
        batch_size=args.batch_size,
        max_len=128
    )

if args.dataset == "BioASQ":
    original_json_path = r"/dataset/BioASQ/trainining14b.json"
    output_path = r"/dataset/BioASQ/processed_bioasq.jsonl"
    process_bioasq(original_json_path, output_path)
    processed_data = load_processed_bioasq(output_path, args.sample_size)
    full_dataset = save_torch_dataset_bioasq(processed_data)
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset_bioasq(
        full_dataset=full_dataset,
        batch_size=args.batch_size,
        max_len=128
    )

if args.dataset == "OBQA":
    input_path = r"/dataset/obqa/train-00000-of-00001.parquet"
    output_path = r"/dataset/obqa/processed_obqa.jsonl"
    process_obqa(input_path, output_path)
    processed_data = load_processed_obqa(output_path, args.sample_size)
    full_dataset = save_torch_dataset_OBQA(processed_data)
    train_loader, val_loader, test_loader = get_loaders_from_existing_dataset_obqa(
        full_dataset=full_dataset,
        batch_size=args.batch_size,
        max_len=128
    )



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
LLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B").to(device)
for param in LLM.parameters():
    param.requires_grad = False

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def train(args, model, train_loader, full_graph, optimizer, criterion, name_to_cuis, text_projector):
    model.train()
    total_loss = 0.0
    train_preds, train_labels = [], []

    for batch in tqdm(train_loader, desc="Training"):
        question_text = batch["input_text"]  # [B, ]
        labels = batch["label"].to(args.device)  # [B, ]
        B = len(question_text)


        batch_keywords = []
        for q in question_text:
            keywords = fast_extract_matched_entities(q, args.index_path)
            batch_keywords.append(list(set(keywords)))

        batch_matched_cuis = map_keywords_to_cuis_new(batch_keywords, name_to_cuis)


        has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
        
        if not has_entities:
            N = 50
            llm_emb_dim = args.llm_emb_dim
            multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
            link_loss = torch.tensor(0.0).to(args.device)
        else:
            batch_subgraphs = batch_get_2hop_subgraph(
                full_graph=full_graph,
                batch_matched_cuis=batch_matched_cuis,
                max_nodes=50
            )


            gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                batch_subgraphs=batch_subgraphs,
                args=args,
                device=args.device,
                tokenizer=tokenizer,
                model=LLM,
                project_layer=text_projector
            )

            multimodal_emb, link_loss = model.graph_grasper(
                gat_inputs=gat_inputs,
                graphs=batch_pyg_graphs,
                text_attributes=batch_text_attrs,
                triples_list=batch_triples
            )


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

        total_batch_loss = cls_loss + (args.link_loss_weight * link_loss if link_loss is not None else 0) + (moe_loss if moe_loss is not None else 0)

        # bp
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # record
        total_loss += total_batch_loss.item() * B
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(train_labels, train_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1

def validate(args, tokenizer, LLM, model, val_loader, full_graph, criterion, name_to_cuis, text_projector):
    model.eval()
    total_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            question_text = batch["input_text"]  # [B, ]
            labels = batch["label"].to(args.device)  # [B, ]
            B = len(question_text)

            batch_keywords = []
            for q in question_text:
                keywords = fast_extract_matched_entities(q, args.index_path)
                batch_keywords.append(list(set(keywords)))

            batch_matched_cuis = map_keywords_to_cuis_new(batch_keywords, name_to_cuis)

            has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
            
            if not has_entities:
                print(f"errpr")
                N = 50
                llm_emb_dim = args.llm_emb_dim
                multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
                link_loss = torch.tensor(0.0).to(args.device)
            else:
                batch_subgraphs = batch_get_2hop_subgraph(
                    full_graph=full_graph,
                    batch_matched_cuis=batch_matched_cuis,
                    max_nodes=50
                )

                gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                    batch_subgraphs=batch_subgraphs,
                    args=args,
                    device=args.device,
                    tokenizer=tokenizer,
                    model=LLM,
                    project_layer=text_projector
                )

                multimodal_emb, link_loss = model.graph_grasper(
                    gat_inputs=gat_inputs,
                    graphs=batch_pyg_graphs,
                    text_attributes=batch_text_attrs,
                    triples_list=batch_triples
                )

            question_embedding = model.question_embeder(question_text)
            the_prompt, moe_loss = model.cross_modality_MHA(multimodal_emb, question_embedding)
            input_embeds = torch.cat([question_embedding, the_prompt], dim=1)
            outputs = model.model(inputs_embeds=input_embeds, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            cls_hidden = hidden_states[:, -1, :]
            logits = model.classifier(cls_hidden)
            cls_loss = criterion(logits, labels)
            total_batch_loss = cls_loss + args.link_loss_weight * link_loss + moe_loss
            total_loss += total_batch_loss.item() * B
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

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
            original_ids = batch["original_id"]  # ID
            B = len(question_text)

            batch_keywords = []
            for q in question_text:
                keywords = fast_extract_matched_entities(q, args.index_path)
                batch_keywords.append(list(set(keywords)))

            batch_matched_cuis = map_keywords_to_cuis_new(batch_keywords, name_to_cuis)

            has_entities = any(len(cuis) > 0 for cuis in batch_matched_cuis)
            
            if not has_entities:
                print(f"errpr")
                N = 50
                llm_emb_dim = args.llm_emb_dim
                multimodal_emb = torch.zeros(B, N, llm_emb_dim).to(args.device)
                link_loss = torch.tensor(0.0).to(args.device)
            else:
                batch_subgraphs = batch_get_2hop_subgraph(
                    full_graph=full_graph,
                    batch_matched_cuis=batch_matched_cuis,
                    max_nodes=50
                )

                gat_inputs, batch_pyg_graphs, batch_text_attrs, batch_triples = batch_umls_graph_to_pyg_data(
                    batch_subgraphs=batch_subgraphs,
                    args=args,
                    device=args.device,
                    tokenizer=tokenizer,
                    model=LLM,
                    project_layer=text_projector
                )

                multimodal_emb, link_loss = model.graph_grasper(
                    gat_inputs=gat_inputs,
                    graphs=batch_pyg_graphs,
                    text_attributes=batch_text_attrs,
                    triples_list=batch_triples
                )


            question_embedding = model.question_embeder(question_text)
            the_prompt, moe_loss = model.cross_modality_MHA(multimodal_emb, question_embedding)
            input_embeds = torch.cat([question_embedding, the_prompt], dim=1)
            outputs = model.model(inputs_embeds=input_embeds, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            cls_hidden = hidden_states[:, -1, :]
            logits = model.classifier(cls_hidden)
            cls_loss = criterion(logits, labels)
            total_batch_loss = cls_loss + args.link_loss_weight * link_loss + moe_loss

            total_loss += total_batch_loss.item() * B
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
            test_original_ids.extend(original_ids)

    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0
    )


    import pandas as pd
    test_results = pd.DataFrame({
        "original_id": test_original_ids,
        "question_text": [q for batch in test_loader for q in batch["input_text"]],
        "true_label": test_labels,
        "pred_label": test_preds,
        "pred_result": ["yes" if p == 1 else "no" for p in test_preds]
    })
    test_results.to_csv("test_results.csv", index=False, encoding="utf-8-sig")
    print(f"to ：test_results.csv")

    return avg_loss, acc, prec, rec, f1


def main(args):
    print(f"args：{args}")

    if args.kgs == "umls":
        try:
            with open(args.graph_path, "rb") as f:
                full_graph = pickle.load(f)
            cui2name_full = {cui: full_graph.nodes[cui]["name"] for cui in full_graph.nodes()}
        except FileNotFoundError:
            print(f"error2")
            return

        name_to_cuis = {}
        for cui, node_name in cui2name_full.items():
            if node_name:
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


    if args.kgs == "NELL":
        with open(args.graph_path, "rb") as f:
            full_graph = pickle.load(f)

        name_to_cuis = {}
        for node_id, attrs in full_graph.nodes(data=True):
            if 'literal_strings' in attrs:
                node_name = attrs['literal_strings']
                if node_name and node_name != 'No literal strings available':
                    name_lower = node_name.lower()
                    if name_lower not in name_to_cuis:
                        name_to_cuis[name_lower] = []
                    name_to_cuis[name_lower].append(node_id)

        with open("global_mapping_NELL.pkl", "rb") as f:
            global_mapping = pickle.load(f)
        global_n_ntype = len(global_mapping["node_type2global_idx"])
        global_n_etype = len(global_mapping["Relation2global_idx"])

        args.n_ntype = global_n_ntype
        args.n_etype = global_n_etype + 1

    text_project_layer = nn.Linear(2560, args.gnn_in_dim).to(device)
    model = MyModel(args, tokenizer, LLM).to(args.device)

    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    for param in text_project_layer.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch{epoch + 1}/{args.epochs} ===")

        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            args, model, train_loader, full_graph, optimizer, criterion, name_to_cuis, text_project_layer
        )
        print(f"train set：Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            args, tokenizer, LLM, model, val_loader, full_graph, criterion, name_to_cuis, text_project_layer
        )
        print(f"val set：Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1
            }, args.save_model_path + ".pth")
            print(f"bestvalF1={best_val_f1:.4f}")

    print(f"\n=== test best ===")
    checkpoint = torch.load(args.save_model_path + ".pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"epoch {checkpoint['epoch']} ,{checkpoint['best_val_f1']:.4f}）")

    test_loss, test_acc, test_prec, test_rec, test_f1 = test(
        args, model, test_loader, full_graph, criterion, name_to_cuis, text_project_layer
    )

    print(f"\result：")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

    print(f"\npath：{args.save_model_path}.pth")


if __name__ == "__main__":
    main(args)
