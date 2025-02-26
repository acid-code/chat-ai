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

# Load smaller dataset for sanity check
# df = pd.read_csv(r"C:\Users\asaf2\Documents\Projects\Psych Model\sainty_check.csv")
# train_data = [{"input": q, "output": a} for q, a in zip(df["question"], df["answer"])]
train_data = [{"input": "מה שלומך?", "output": "טוב, תודה ששאלת. איך אני יכול לעזור?"}]

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
    print("Tokenized inputs:", inputs)
    print("Tokenized outputs:", outputs)
    return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}


from datasets import Dataset

# Create dataset
dataset = Dataset.from_list(train_data).map(tokenize_data)

# Print a few tokenized samples to verify tokenization
print("Sample tokenized data:", dataset[:3])

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Increase the number of epochs for better training
    per_device_train_batch_size=2,
    learning_rate=5e-5,  # Adjust the learning rate
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
model.save_pretrained(r"C:\Users\asaf2\Documents\Projects\Psych Model\sainty_model")
tokenizer.save_pretrained(r"C:\Users\asaf2\Documents\Projects\Psych Model\sainty_model")
print("Model saved successfully!")


def debug_model_output(text):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        print("Input IDs:", input_ids)

        outputs = model.generate(input_ids)
        print("Raw Outputs:", outputs)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Decoded Response:", response)
    except Exception as e:
        print(f"Error during model output debugging: {e}")


# Test the model with a sample input
debug_model_output("מה שלומך?")
