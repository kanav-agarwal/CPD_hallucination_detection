"""
download_dataset.py

Downloads the SQuAD v2 validation set, filters for
unanswerable questions, formats prompts, and saves them
to outputs/prompts.json.

Paper:
Variance in Token Log Probabilities as a Hallucination Signal
"""

from pathlib import Path
import json

from datasets import load_dataset


# ----------------------------
# Settings
# ----------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "prompts.json"

# Prompt template used throughout the project.
# (We'll modify this later if we discover the authors
# used a slightly different wording.)

PROMPT_TEMPLATE = """Answer the following question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""


# ----------------------------
# Download Dataset
# ----------------------------

print("Downloading SQuAD v2...")

dataset = load_dataset(
    "rajpurkar/squad_v2",
    split="validation"
)

print(f"Downloaded {len(dataset)} validation examples.")


# ----------------------------
# Keep only impossible questions
# ----------------------------

dataset = dataset.filter(
    lambda x: len(x["answers"]["text"]) == 0
)

print(f"Keeping {len(dataset)} impossible questions.")


# ----------------------------
# Format prompts
# ----------------------------

prompts = []

for example in dataset:

    prompt = PROMPT_TEMPLATE.format(
        context=example["context"],
        question=example["question"]
    )

    prompts.append({

        "id": example["id"],

        "title": example["title"],

        "context": example["context"],

        "question": example["question"],

        "prompt": prompt

    })


# ----------------------------
# Save
# ----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        prompts,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSaved {len(prompts)} prompts.")
print(f"Location: {OUTPUT_FILE}")