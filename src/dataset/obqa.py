
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.lm_modeling import load_model, load_text2embedding

PATH = 'dataset/obqa'

class OBQADataset(Dataset):
    def __init__(self):
        super().__init__()
        import pandas as pd
        self.text = pd.read_json(f'{PATH}/all_data.jsonl', lines=True)
        self.prompt = 'Answer the question by choosing the correct option label (A, B, C, D).'
        self.graph_type = 'ConceptNet Subgraph'

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        item = self.text.iloc[index]
        graph = torch.load(f'{PATH}/graphs/{index}.pt', weights_only=False)
        
        choices = item['choices']
        
        choice_str = ""
        # Handle choices being either a dict or a string representation of a dict
        if isinstance(choices, str):
            import ast
            try:
                choices = ast.literal_eval(choices)
            except:
                pass
                
        if isinstance(choices, dict):
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            for l, t in zip(labels, texts):
                choice_str += f"{l}: {t}\n"
        
        question = f"Question: {item['question_stem']}\nChoices:\n{choice_str}\n{self.prompt}"
        
        # Load nodes/edges description for retrieval if needed (G-Retriever uses it)
        try:
            nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
            edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
            # Limit description length to avoid OOM or too long context
            desc = "Nodes:\n" + nodes.head(20).to_csv(index=False) + "\nEdges:\n" + edges.head(20).to_csv(index=False)
        except:
            desc = ""

        return {
            'id': index,
            'label': item['answerKey'],
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
