from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import json
import torch
import os
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# Load dataset
dataset = load_dataset(
    "Shenlab/MentalChat16K", split="train[:700]"
)  # Change split if needed

system_instruction = "You are a professional psychiatrist conducting a therapy session. Be empathetic, logical, and structured in your response."

# Process and format the dataset
formatted_data = []
for item in dataset:
    if not item or not item["input"]:
        continue
    input_text = item["input"].strip()
    output_text = item["output"].strip()

    # Format for OpenLLaMA
    formatted_entry = {
        "input": f"<|system|>\n{system_instruction}</s>\n<|user|>\n{input_text}</s>\n<|assistant|>\n{output_text}",
        "output": output_text,  # Keep raw output for reference
    }
    formatted_data.append(formatted_entry)

# Save as JSONL
output_file = "datasets/tinyllama_formatted_mentalchat16k.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset saved to {output_file}")

# Model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix missing padding token

# Load dataset
dataset = load_dataset(
    "json", data_files="datasets/tinyllama_formatted_mentalchat16k.jsonl", split="train"
)


# Tokenization function
def tokenize_function(example):
    tokenized = tokenizer(
        example["input"],
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"][:]  # Copy input_ids to labels
    return tokenized


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# LoRA Configuration for QLoRA
lora_config = LoraConfig(
    r=16,  # Low-rank adaptation
    lora_alpha=32,  # Alpha scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,  # Dropout for stability
    bias="none",  # No extra biases
    task_type="CAUSAL_LM",  # Type of task (Language Modeling)
)


# Load model with proper device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use 16-bit for lower RAM usage
    device_map="auto",  # Automatically assign layers to GPU/CPU
)

accelerator = Accelerator()
model = get_peft_model(model, lora_config)
model, tokenized_dataset = accelerator.prepare(model, tokenized_dataset)

# Training Arguments (Optimized for Colab GPU)
training_args = TrainingArguments(
    output_dir="./finetuned_tinyllama",
    evaluation_strategy="no",
    logging_steps=10,
    num_train_epochs=2,
    learning_rate=2e-5,
    save_total_limit=2,
)

# Update Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

# Resume training if checkpoint exists
last_checkpoint = None
if os.path.isdir("./finetuned_tinyllama") and any(
    fname.startswith("checkpoint") for fname in os.listdir("./finetuned_tinyllama")
):
    last_checkpoint = sorted(
        [f for f in os.listdir("./finetuned_tinyllama") if f.startswith("checkpoint")]
    )[-1]
    last_checkpoint = f"./finetuned_tinyllama/{last_checkpoint}"

trainer.train(resume_from_checkpoint=last_checkpoint)

# Save model
model.save_pretrained("models/finetuned_tinyllama")
tokenizer.save_pretrained("models/finetuned_tinyllama")

model.push_to_hub(
    "finetuned_tinyllama", use_auth_token="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"
)
tokenizer.push_to_hub(
    "finetuned_tinyllama", use_auth_token="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"
)
