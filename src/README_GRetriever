# G-Retriever with OpenBookQA and PubMedQA

This project has been extended to support OpenBookQA (with ConceptNet) and PubMedQA (with UMLS), and now supports **Qwen3-4B** as a backbone model.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install pandas transformers datasets torch_geometric networkx tqdm accelerate peft
    ```

2.  **Generate Knowledge Graphs**:
    For testing, we use mock KGs. For real experiments, replace `dataset/kg/mock_cpnet.pkl` and `dataset/kg/mock_umls.pkl` with real graphs.
    ```bash
    python dataset/kg/create_mock_kgs.py
    ```

3.  **Preprocess Data**:
    This step loads the datasets, retrieves subgraphs from the KG, and encodes them.
    ```bash
    python -m src.dataset.preprocess.obqa
    python -m src.dataset.preprocess.pubmedqa
    ```
    *Note: In the current test environment, this uses dummy data and mock embeddings.*

## Training

To train G-Retriever on these datasets:

**OpenBookQA with Qwen3-4B**:
```bash
python train.py --dataset obqa --model_name graph_llm --num_epochs 5 --llm_model_name qwen3-4b --gnn_in_dim 384
```

**PubMedQA with Qwen3-4B**:
```bash
python train.py --dataset pubmedqa --model_name graph_llm --num_epochs 5 --llm_model_name qwen3-4b --gnn_in_dim 384
```

**Using a Local Model**:
If you have downloaded the model to a local directory (e.g., `/data/models/Qwen3-4B`), you can pass the path directly:
```bash
python train.py --dataset obqa --model_name graph_llm --llm_model_name /data/models/Qwen3-4B --gnn_in_dim 384
```

## Handling Network Issues

If you encounter `ConnectTimeout` errors (common in restricted network environments):
1.  **Automatic Fallback**: The code has been updated to automatically switch to a dummy mock model if a network timeout occurs. This allows you to verify the pipeline functionality without downloading the full model.
2.  **Manual Download**: Download the model manually from Hugging Face (or a mirror) and use the local path as shown above.

## Configuration Notes for Real Performance

To get real performance results, you need to revert the changes made for the test environment:

1.  **SBERT**: In `src/utils/lm_modeling.py`, revert `load_sbert` to use `sentence-transformers/all-roberta-large-v1` and remove the mock implementation.
2.  **LLM**: In `src/model/__init__.py`, change `llama_model_path` back to `meta-llama/Llama-2-7b-hf` (or use `qwen3-4b` which points to `Qwen/Qwen3-4B`).
3.  **Data**: In `src/dataset/preprocess/obqa.py` and `pubmedqa.py`, enable `load_dataset` and remove dummy data fallback.
4.  **GNN Input Dim**: When using real SBERT (dim 1024/768/384?), ensure `--gnn_in_dim` matches the embedding size. (Default SBERT is 384 for mini, 768 for base, 1024 for large).
