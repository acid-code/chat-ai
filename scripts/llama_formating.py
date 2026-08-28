from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("Shenlab/MentalChat16K", split="train")  # Change split if needed

# Process and format the dataset
formatted_data = []
for item in dataset:
    if not item["input"]:
        continue
    input_text = item["input"].strip()
    output_text = item["output"].strip()

    # Format for OpenLLaMA
    formatted_entry = {
        "input": f"<s> User: {input_text}\nAssistant: {output_text} </s>",
        "output": output_text,  # Keep raw output for reference
    }
    formatted_data.append(formatted_entry)

print(dataset[:1])
# Save as JSONL
output_file = "datasets/formatted_mentalchat16k.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset saved to {output_file}")
