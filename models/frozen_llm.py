from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def CallFrozenModel(model_name="Qwen/Qwen2.5-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    return tokenizer, model