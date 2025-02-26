from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import torch
import pandas as pd

# Load tokenizer and model
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# import intel_npu_acceleration_library
# from intel_npu_acceleration_library.compiler import CompilerConfig

# compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
# model = intel_npu_acceleration_library.compile(model, compiler_conf)

# Load dataset
df = pd.read_csv(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\counselchat_hebrew3.csv"
)
train_data = [{"input": q, "output": a} for q, a in zip(df["question"], df["answer"])]

for item in train_data:
    item["input"] = str(item["input"])
    item["output"] = str(item["output"])


# Tokenize data
def tokenize_data(examples):
    inputs = tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=256
    )
    outputs = tokenizer(
        examples["output"], padding="max_length", truncation=True, max_length=256
    )
    return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}


from datasets import Dataset

# Create dataset
dataset = Dataset.from_list(train_data).map(tokenize_data)

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\hebrew_counseling_model"
)
tokenizer.save_pretrained(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\hebrew_counseling_model"
)
print("Model saved successfully!")
