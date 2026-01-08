# UPer: Unified Prompter for Language Models via Knowledge Graphs

This repo provide the source code for paper "UPer: Unified Prompter for Large Language Models via Knowledge Graphs".

Simple comparison:

<img width="548" height="294" alt="F65D0462D5B0283C3FC9C06BAC77315D" src="https://github.com/user-attachments/assets/f6850123-8998-49b9-ac87-23b7b8cae0d6" />

## Overview
Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks but suffer from critical flaws like hallucinations and inadequate complex reasoning, stemming from insufficient grounding in factual knowledge. Knowledge Graphs (KGs), as structured repositories containing entities, relations, and rich textual attributes, offer a solution to enhance grounded knowledge of LLM, yet existing KG-LLM integration methods only leverage KG topological structures, neglecting other valuable information like text attributes. To address this gap, we propose UnifiedPrompter (UPer), a framework that fully harnesses KGs’ comprehensive information (topological and textual) to augment pre-trained LLMs. To address hallucinations and complex reasoning, UPer consists of two core components: Graph Grasper aims to fully capture and extract information from KGs, whereas Knowledge-aware Attention is dedicated to finding the most relevant segment for downstream tasks. Experimental results across six benchmark datasets for commonsense and biomedical reasoning tasks show that UPer outperforms the Graph Neural Prompting method by an average of 8.31\%

<img width="950" height="533" alt="7871DB482A2E77496855E7C8820B8D61" src="https://github.com/user-attachments/assets/5d7d34f1-e87f-45e6-96d6-e873180842a2" />

## 1.Denpendencies
```
pip install -r requirements.txt
```

## 2.Run UPer
To run UPer on various datasets, simply use instructions as follows:
```
python train.py --dataset --kgs
```
`--dataset` includes BioASQ, PQA, OBQA, Riddle, ARC, PIQA
`--kgs` includes umls and NELL
