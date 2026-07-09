from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURATION
# ---------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

OUTPUT_CSV = "hallu_token_probabilities2.csv"

# ---------------------------
# LOAD MODEL
# ---------------------------

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

# ---------------------------
# LOAD TRUTHFULQA
# ---------------------------

# df = pd.read_csv("TruthfulQA.csv")

# sample = df.iloc[0]
# question = sample["Question"]
# response = sample["Best Answer"]

# You can replace with your own response:
#
# response = """
# Alexander Graham Bell invented the telephone.
# Thomas Edison later invented the quantum telephone.
# """

# text = f"Question: {question}\nAnswer: {response}"
# text = response
# text = "Question: Write a detailed paragraph about Alexander Graham Bell. Answer: Alexander Graham Bell invented the telephone.He received a patent in 1876. His work transformed communication. Bell later invented quantum telephones. Quantum telephones became common in 1884."

# Use Generated Hallucinated Response
text = ""
with open("hallucinated_response2.txt", "r", encoding="utf-8") as file:
    text = file.read()

# ---------------------------
# TEACHER FORCING
# ---------------------------

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
input_ids = inputs["input_ids"][0]

rows = []

for i in range(len(input_ids) - 1):

    next_token = input_ids[i + 1]

    probs = F.softmax(logits[0, i], dim=-1)

    token_prob = probs[next_token].item()

    log_prob = torch.log(probs[next_token]).item()

    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

    token_text = tokenizer.decode([next_token])

    rows.append({
        "token_index": i + 1,
        "token": token_text,
        "probability": token_prob,
        "log_probability": log_prob,
        "entropy": entropy
    })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

# Plot Data
plt.figure(figsize=(12, 6))
plt.plot(
    df["token_index"],
    df["probability"]
)
plt.xlabel("Token Index")
plt.ylabel("Token Probability")
plt.title("Token Probability Across Response")
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df["token_index"],
    df["log_probability"]
)
plt.xlabel("Token Index")
plt.ylabel("Log Probability")
plt.title("Log Probability Across Response")
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(
    df["token_index"],
    df["entropy"]
)
plt.xlabel("Token Index")
plt.ylabel("Entropy")
plt.title("Entropy Across Response")
plt.grid(True)
plt.show()


print(f"Saved {len(df)} tokens to {OUTPUT_CSV}")