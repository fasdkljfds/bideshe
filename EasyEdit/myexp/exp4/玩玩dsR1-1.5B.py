from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import json

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to("cuda")  # Move model to GPU

model.generate(
    torch.tensor(
        tokenizer.encode("请问你是谁", return_tensors="pt")
    ).to("cuda"),
    do_sample=True,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=5,
    use_cache=True,
)

print('caonima'')
# hahah