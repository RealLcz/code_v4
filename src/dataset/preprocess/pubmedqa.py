
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import pickle
import networkx as nx
from datasets import load_dataset
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/pubmedqa'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def get_subgraph(G, text, hops=2):
    # Simple entity linking
    tokens = text.lower().split()
    seeds = [n for n in G.nodes() if str(n).lower() in tokens]
    
    if not seeds:
        if len(G.nodes()) > 0:
            seeds = [list(G.nodes())[0]]
        else:
            return pd.DataFrame(columns=['node_id', 'node_attr']), pd.DataFrame(columns=['src', 'edge_attr', 'dst'])

    subgraph_nodes = set(seeds)
    current_layer = set(seeds)
    
    for _ in range(hops):
        next_layer = set()
        for node in current_layer:
            if node in G:
                neighbors = list(G.neighbors(node))
                next_layer.update(neighbors)
        subgraph_nodes.update(next_layer)
        current_layer = next_layer
        
    subgraph = G.subgraph(subgraph_nodes)
    
    nodes = []
    edges = []
    node_map = {}
    
    for i, node in enumerate(subgraph.nodes()):
        node_map[node] = i
        nodes.append({'node_id': i, 'node_attr': str(node)})
        
    for u, v, data in subgraph.edges(data=True):
        if u in node_map and v in node_map:
            edges.append({
                'src': node_map[u],
                'edge_attr': str(data.get('rel_id', 'related_to')),
                'dst': node_map[v]
            })
            
    return pd.DataFrame(nodes), pd.DataFrame(edges)

def step_one():
    print("Loading PubMedQA...")
    try:
        dataset = load_dataset("Datasaur/PubMedQA-llama-405B-labels")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Using dummy data.")
        dataset = {
            'train': [{'context': {'contexts': ['This is context.']}, 'question': 'Is it true?', 'final_decision': 'yes'} for _ in range(20)]
        }
    
    print("Dataset loaded successfully.")
    
    full_data = []
    
    # Check for splits
    splits = ['train', 'validation', 'test']
    has_splits = False
    for split in splits:
        if split in dataset:
            has_splits = True
            for item in dataset[split]:
                if not isinstance(item, dict):
                    item = dict(item)
                item['split'] = split
                full_data.append(item)
    
    # If no splits found (e.g. only 'train' exists like original pqa_labeled), manually split
    if not has_splits or (len(dataset) == 1 and 'train' in dataset):
        print("No explicit validation/test splits found or only train split available. Splitting manually...")
        # If we already populated full_data from 'train' above, we just need to re-assign 'split'
        # But if 'train' was the only split, full_data has all of them with split='train'
        
        if not full_data and 'train' in dataset:
             for item in dataset['train']:
                if not isinstance(item, dict):
                    item = dict(item)
                full_data.append(item)

        # Simple split 80/10/10
        n = len(full_data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        for i in range(n):
            if i < train_end:
                full_data[i]['split'] = 'train'
            elif i < val_end:
                full_data[i]['split'] = 'validation'
            else:
                full_data[i]['split'] = 'test'
            
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)
    
    print("Loading Knowledge Graph...")
    with open('dataset/kg/mock_umls.pkl', 'rb') as f:
        kg = pickle.load(f)

    print("Extracting subgraphs...")
    for i, item in tqdm(enumerate(full_data), total=len(full_data)):
        # Handle Datasaur/PubMedQA-llama-405B-labels schema
        # Columns: prompt, long_answer, final_decision
        
        if 'prompt' in item:
            text = item['prompt']
        else:
            # Fallback to old schema logic if 'prompt' not found
            context_str = ""
            context_obj = item.get('context', "")
            
            if isinstance(context_obj, str):
                context_str = context_obj
            elif isinstance(context_obj, list):
                context_str = " ".join(context_obj)
            elif isinstance(context_obj, dict):
                # Try 'contexts' key (standard pqa) or just values
                if 'contexts' in context_obj:
                    c_val = context_obj['contexts']
                    if isinstance(c_val, list):
                        context_str = " ".join(c_val)
                    else:
                        context_str = str(c_val)
                else:
                    context_str = str(context_obj)
            
            text = str(item.get('question', '')) + " " + context_str

        nodes, edges = get_subgraph(kg, text)
        
        if len(nodes) > 0:
            nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        else:
            # Use empty list for columns to avoid issues
            pd.DataFrame(columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{i}.csv', index=False)
            
        if len(edges) > 0:
            edges.to_csv(f'{path_edges}/{i}.csv', index=False)
        else:
            pd.DataFrame(columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{i}.csv', index=False)

    # Save split indices
    os.makedirs(f'{path}/split', exist_ok=True)
    train_indices = [i for i, item in enumerate(full_data) if item.get('split') == 'train']
    val_indices = [i for i, item in enumerate(full_data) if item.get('split') == 'validation']
    test_indices = [i for i, item in enumerate(full_data) if item.get('split') == 'test']
    
    with open(f'{path}/split/train_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, train_indices)))
    with open(f'{path}/split/val_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, val_indices)))
    with open(f'{path}/split/test_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, test_indices)))

    # Save data
    # Convert numpy types to python types for json serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Clean data before saving
    cleaned_data = []
    for item in full_data:
        cleaned_item = {k: convert_to_serializable(v) for k, v in item.items()}
        cleaned_data.append(cleaned_item)
        
    pd.DataFrame(cleaned_data).to_json(f'{path}/all_data.jsonl', orient='records', lines=True)

def step_two():
    print("Encoding graphs...")
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]
    
    os.makedirs(path_graphs, exist_ok=True)
    data_df = pd.read_json(f'{path}/all_data.jsonl', lines=True)
    
    for i in tqdm(range(len(data_df))):
        try:
            nodes = pd.read_csv(f'{path_nodes}/{i}.csv')
            edges = pd.read_csv(f'{path_edges}/{i}.csv')
            
            if len(nodes) == 0:
                x = torch.zeros(0, 384)
                edge_index = torch.LongTensor([[], []])
                e = torch.zeros(0, 384)
                num_nodes = 0
            else:
                x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
                if len(edges) > 0:
                    e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
                    edge_index = torch.LongTensor([edges.src, edges.dst])
                else:
                    e = torch.zeros(0, 384)
                    edge_index = torch.LongTensor([[], []])
                num_nodes = len(nodes)
                
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=num_nodes)
            torch.save(data, f'{path_graphs}/{i}.pt')
        except Exception as e:
            print(f"Error at index {i}: {e}")
            data = Data(x=torch.zeros(0, 384), edge_index=torch.LongTensor([[], []]), edge_attr=torch.zeros(0, 384), num_nodes=0)
            torch.save(data, f'{path_graphs}/{i}.pt')

if __name__ == '__main__':
    step_one()
    step_two()
