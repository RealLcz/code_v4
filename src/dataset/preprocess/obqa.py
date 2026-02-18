
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
path = 'dataset/obqa'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def get_subgraph(G, question, hops=2):
    # Simple entity linking: find nodes that appear in the question
    q_tokens = question.lower().split()
    seeds = [n for n in G.nodes() if str(n).lower() in q_tokens]
    
    if not seeds:
        # If no seeds found, pick random nodes just to have a graph (for mock purposes)
        if len(G.nodes()) > 0:
            seeds = [list(G.nodes())[0]]
        else:
            return [], []

    subgraph_nodes = set(seeds)
    current_layer = set(seeds)
    
    for _ in range(hops):
        next_layer = set()
        for node in current_layer:
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
    print("Loading OpenBookQA...")
    try:
        dataset = load_dataset("allenai/openbookqa", "main")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Using dummy data.")
        dataset = {
            'train': [{'question_stem': 'Apple is red?', 'choices': {'text': ['Yes', 'No', 'Maybe', 'Unknown'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'A'} for _ in range(10)],
            'validation': [{'question_stem': 'Banana is yellow?', 'choices': {'text': ['Yes', 'No', 'Maybe', 'Unknown'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'A'} for _ in range(5)],
            'test': [{'question_stem': 'Sky is blue?', 'choices': {'text': ['Yes', 'No', 'Maybe', 'Unknown'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'A'} for _ in range(5)]
        }

    print("Dataset loaded successfully.")
    # Combine train/val/test for processing
    all_data = []
    
    # Handle both DatasetDict and manual dictionary from fallback
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        if split in dataset:
            for item in dataset[split]:
                # Convert to dict if it's a Dataset item
                if not isinstance(item, dict):
                    item = dict(item)
                item['split'] = split
                all_data.append(item)
            
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)
    
    print("Loading Knowledge Graph...")
    with open('dataset/kg/mock_cpnet.pkl', 'rb') as f:
        kg = pickle.load(f)

    print("Extracting subgraphs...")
    valid_indices = []
    for i, item in tqdm(enumerate(all_data), total=len(all_data)):
        question = item['question_stem']
        nodes, edges = get_subgraph(kg, question)
        
        # Ensure directory existence for each batch/item if not already created globally
        # (Already created above, but good practice to be safe)
        
        if len(nodes) > 0 and len(edges) > 0:
            nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
            edges.to_csv(f'{path_edges}/{i}.csv', index=False)
            valid_indices.append(i)
        else:
            # Create empty placeholder
            # Use empty list for columns to avoid "No columns to parse from file" later if we don't write header
            # But here we write header.
            pd.DataFrame(columns=['node_id', 'node_attr']).to_csv(f'{path_nodes}/{i}.csv', index=False)
            pd.DataFrame(columns=['src', 'edge_attr', 'dst']).to_csv(f'{path_edges}/{i}.csv', index=False)
            valid_indices.append(i) # Keep index to align with data

    # Save split indices
    os.makedirs(f'{path}/split', exist_ok=True)
    train_indices = [i for i, item in enumerate(all_data) if item['split'] == 'train']
    val_indices = [i for i, item in enumerate(all_data) if item['split'] == 'validation']
    test_indices = [i for i, item in enumerate(all_data) if item['split'] == 'test']
    
    with open(f'{path}/split/train_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, train_indices)))
    with open(f'{path}/split/val_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, val_indices)))
    with open(f'{path}/split/test_indices.txt', 'w') as f:
        f.write('\n'.join(map(str, test_indices)))

    # Save the text data for easy loading later
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
    for item in all_data:
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
                # Empty graph
                x = torch.zeros(0, 384) # SBERT dim
                edge_index = torch.LongTensor([[], []])
                edge_attr = torch.zeros(0, 384)
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
            # Save empty
            data = Data(x=torch.zeros(0, 384), edge_index=torch.LongTensor([[], []]), edge_attr=torch.zeros(0, 384), num_nodes=0)
            torch.save(data, f'{path_graphs}/{i}.pt')

if __name__ == '__main__':
    step_one()
    step_two()
