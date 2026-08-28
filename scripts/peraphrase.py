from transformers import pipeline
import json

# Load a paraphrase model (T5, Pegasus, etc.)
paraphraser = pipeline("text2text-generation", model="t5-small")

# Load existing dataset
with open("../datasets/classified_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Generate paraphrases
expanded_data = []
logging_interval = 300
for i, entry in enumerate(data):
    expanded_data.append(entry)
    current_entry = entry
    for _ in range(2):
        if (
            len('{current_entry["question"]}') > 250
            or len('{current_entry["answer"]}') > 250
        ):
            continue
        input_variation = paraphraser(
            f"Paraphrase: {current_entry['question']}", max_length=256
        )
        output_variation = paraphraser(
            f"Paraphrase: {current_entry['answer']}", max_length=256
        )
        new_entry = current_entry.copy()
        new_entry["question"] = input_variation[0]["generated_text"][2:]
        new_entry["answer"] = output_variation[0]["generated_text"][2:]
        expanded_data.append(new_entry)
        current_entry = new_entry
    if i % logging_interval == 0:
        print(f"Processed {i} entries")

# Save expanded dataset
with open("../datasets/expanded_dataset.json", "w", encoding="utf-8") as f:
    json.dump(expanded_data, f, indent=4, ensure_ascii=False)
