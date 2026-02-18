from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "sshleifer/tiny-gpt2",
    "7b_chat": "sshleifer/tiny-gpt2",
    "13b": "sshleifer/tiny-gpt2",
    "13b_chat": "sshleifer/tiny-gpt2",
    "qwen3-4b": "Qwen/Qwen3-4B",
}
