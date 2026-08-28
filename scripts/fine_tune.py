from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import torch
import pandas as pd

# Load tokenizer and model from the saved checkpoint
model_path = (
    r"C:\Users\asaf2\Documents\Projects\Psych Model\models\hebrew_counseling_model"
)
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# Load dataset
df = pd.read_csv(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\datasets\counselchat_hebrew3.csv"
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
    num_train_epochs=3,  # You can increase the number of epochs for further fine-tuning
    per_device_train_batch_size=2,
    learning_rate=1e-4,  # Adjust the learning rate if needed
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Continue training the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\models\hebrew_counseling_model_finetuned"
)
tokenizer.save_pretrained(
    r"C:\Users\asaf2\Documents\Projects\Psych Model\models\hebrew_counseling_model_finetuned"
)
print("Model fine-tuned and saved successfully!")
