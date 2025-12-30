"""
Script to process PIQA dataset
This script combines the goal with each solution and adds labels based on the labels file.

Input:
- train.jsonl: Contains goal and two solutions for each example
- train-labels.lst: Contains labels (0 or 1) where:
  - 0 means sol1 is the correct solution
  - 1 means sol2 is the correct solution

Output:
- processed_piqa.jsonl: Each line contains:
  {
    "id": "example_id",
    "sentence1": "goal + sol1",
    "sentence2": "goal + sol2",
    "label1": 0 or 1,  # 1 if sol1 is correct, else 0
    "label2": 0 or 1   # 1 if sol2 is correct, else 0
  }
"""

import json
import os
from tqdm import tqdm

def process_piqa(jsonl_path, labels_path, output_path):
    """Process PIQA dataset by combining goal with solutions and adding labels."""
    print(f'Processing PIQA dataset...')
    print(f'Input JSONL: {jsonl_path}')
    print(f'Labels file: {labels_path}')
    print(f'Output: {output_path}')
    
    # Read labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f]
    
    # Read and process JSONL file
    processed = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=len(labels))):
            if i >= len(labels):
                print(f"Warning: More examples in JSONL than labels. Stopping at example {i}.")
                break
            
            # Parse JSON
            data = json.loads(line.strip())
            
            # Combine goal with solutions
            goal = data['goal'].strip()
            sol1 = data['sol1'].strip()
            sol2 = data['sol2'].strip()
            
            # Handle punctuation to make the combined sentence grammatically correct
            # Check if goal ends with punctuation
            if not goal.endswith(('.', '?', '!', ',', ':')):
                # If goal is a question and solution starts with a capital letter, add a space
                # Otherwise, add a space
                sentence1 = f"{goal} {sol1}"
                sentence2 = f"{goal} {sol2}"
            else:
                # If goal ends with punctuation, just add a space
                sentence1 = f"{goal} {sol1}"
                sentence2 = f"{goal} {sol2}"
            
            # Get label
            label = labels[i]
            
            # Create processed entry
            processed_entry1 = {
                "id": data['id'],
                "sentence": sentence1,
                "label": 1 - label,  # 1 if sol1 is correct
            }
            
            processed.append(processed_entry1)

            processed_entry2 = {
                "id": data['id'],
                "sentence": sentence2,
                "label": label
            }
            processed.append(processed_entry2)
    
    # Write processed data to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write('\n')
    
    print(f'Processing complete.')
    print(f'Processed {len(processed)} examples.')
    print(f'Output saved to: {output_path}')

if __name__ == '__main__':
    # Default paths
    jsonl_path = r"c:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\PIQA\physicaliqa-train-dev\train.jsonl"
    labels_path = r"c:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\PIQA\physicaliqa-train-dev\train-labels.lst"
    output_path = r"c:\Users\30975\PycharmProjects\pythonProject1\Gnprompter_for_Linux\dataset\PIQA\physicaliqa-train-dev\processed_piqa.jsonl"
    
    process_piqa(jsonl_path, labels_path, output_path)