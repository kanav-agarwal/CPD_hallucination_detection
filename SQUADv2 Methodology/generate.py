import os
import json
import torch
import pandas as pd

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "EleutherAI/gpt-neo-125M"

PROMPT_FILE = "outputs/prompts.json"

OUTPUT_DIR = "outputs/token_data"

NUM_PROMPTS = 10

MAX_NEW_TOKENS = 80

TEMPERATURE = 1.2

TOP_P = 0.95

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load prompts
# ==========================================================

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = json.load(f)

prompts = prompts[:NUM_PROMPTS]

print(f"Loaded {len(prompts)} prompts.")

# ==========================================================
# Load model
# ==========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading GPT-Neo...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)

model.eval()

print("Model loaded.\n")

# ==========================================================
# Generate
# ==========================================================

for prompt_idx, sample in enumerate(tqdm(prompts)):

    prompt = sample["prompt"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            return_dict_in_generate=True,

            output_scores=True,

            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs.sequences[
        0,
        inputs.input_ids.shape[1]:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("=" * 80)
    print(f"Prompt {prompt_idx}")
    print("=" * 80)
    print(sample["question"])
    print()
    print(response)
    print()

    rows = []

    for position, (token_id, logits) in enumerate(
        zip(generated_ids, outputs.scores)
    ):

        probs = torch.softmax(
            logits[0],
            dim=-1
        )

        probability = probs[token_id].item()

        log_probability = torch.log(
            probs[token_id]
        ).item()

        entropy = -torch.sum(
            probs * torch.log(probs + 1e-12)
        ).item()

        token = tokenizer.decode(token_id)

        rows.append({

            "position": position,

            "token": token,

            "token_id": token_id.item(),

            "probability": probability,

            "log_probability": log_probability,

            "entropy": entropy

        })

    df = pd.DataFrame(rows)

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"prompt_{prompt_idx}.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    txt_file = os.path.join(
        OUTPUT_DIR,
        f"prompt_{prompt_idx}.txt"
    )

    with open(txt_file, "w", encoding="utf-8") as f:

        f.write(prompt)

        f.write("\n\n")

        f.write("=" * 80)

        f.write("\nGenerated Response\n\n")

        f.write(response)

print("\nDone.")