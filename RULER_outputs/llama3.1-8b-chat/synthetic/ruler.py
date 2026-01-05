import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).eval()

from datasets import load_dataset
ds = load_dataset("rbiswasfc/ruler", split="train")  # :contentReference[oaicite:4]{index=4}
print(ds.features)
print(ds[0].keys(), ds[0])