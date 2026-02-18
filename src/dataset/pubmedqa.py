
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.lm_modeling import load_model, load_text2embedding

PATH = 'dataset/pubmedqa'

class PubMedQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.text = pd.read_json(f'{PATH}/all_data.jsonl', lines=True)
        self.prompt = 'Answer the question with yes, no, or maybe.'
        self.graph_type = 'UMLS Subgraph'

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        item = self.text.iloc[index]
        graph = torch.load(f'{PATH}/graphs/{index}.pt', weights_only=False)
        
        # Check if 'prompt' exists (Datasaur/PubMedQA-llama-405B-labels)
        if 'prompt' in item and pd.notna(item['prompt']):
            question = item['prompt']
            if self.prompt not in question:
                question += f"\n{self.prompt}\nAnswer:"
        else:
            # Fallback to standard schema
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
            
            question = f"Context: {context_str}\nQuestion: {item['question']}\n{self.prompt}\nAnswer:"
        
        # Robust label extraction
        label = item.get('final_decision')
        if label is None:
            label = item.get('answer')
        if label is None:
            label = item.get('label')
            
        try:
            nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
            edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
            # Limit description length
            desc = "Nodes:\n" + nodes.head(20).to_csv(index=False) + "\nEdges:\n" + edges.head(20).to_csv(index=False)
        except:
            desc = ""

        return {
            'id': index,
            'label': label,
            'desc': desc,
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file if line.strip()]
        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file if line.strip()]
        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file if line.strip()]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}
