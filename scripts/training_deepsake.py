import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load model and tokenizer
model_name = "mediocredev/open-llama-3b-v2-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset(
    "json", data_files="datasets/formatted_mentalchat16k.jsonl", split="train"
)


# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["input"], padding="max_length", truncation=True, max_length=512
    )


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA Configuration for QLoRA
lora_config = LoraConfig(
    r=16,  # Low-rank adaptation
    lora_alpha=32,  # Alpha scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,  # Dropout for stability
    bias="none",  # No extra biases
    task_type="CAUSAL_LM",  # Type of task (Language Modeling)
)

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Load model in 4-bit for efficiency
    torch_dtype=torch.float16,  # Use 16-bit for lower RAM usage
    device_map="auto",  # Auto-select device (CPU or GPU if available)
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./finetuned_openllama",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    per_device_train_batch_size=1,  # Keep batch size small for CPU
    gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger batch
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision
    optim="paged_adamw_8bit",  # Optimized for low RAM
    save_total_limit=2,
    push_to_hub=False,  # Change to True if you want to upload
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model
trainer.train()

# Save final model
model.save_pretrained("models/finetuned_openllama")
tokenizer.save_pretrained("models/finetuned_openllama")
