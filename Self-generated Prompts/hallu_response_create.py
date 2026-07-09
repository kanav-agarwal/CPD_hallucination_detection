from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = """
You are writing a Wikipedia-style article.

Write a 700-word biography of Stephen Hawking.

Rules:
- Only output the article.
- No meta commentary.
- No instructions.
- No formatting guidance.
- No explanations of structure.
- No bullet points or lists.

The article must include realistic-sounding:
- institutions
- dates
- collaborations
- research contributions

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=1.5,
    top_p=0.95,
    repetition_penalty=1.1
)

response = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print(response)

with open("hallucinated_response4.txt", "w", encoding="utf-8") as f:
    f.write(response)