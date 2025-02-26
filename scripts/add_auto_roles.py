from transformers import pipeline
import pandas as pd
import json

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("created classifier")
# Define possible roles
roles = ["therapist", "friend", "humorist", "coach"]

df = pd.read_csv("../datasets/counselchat_english.csv")
print("loaded db")
# Assign roles
classified_data = []
for _, entry in df.iterrows():
    role = classifier(entry["question"], roles)["labels"][0]  # Most likely role
    entry["role"] = role
    classified_data.append(entry.to_dict())

print("added classification")
# Save updated dataset
with open("../datasets/classified_dataset.json", "w", encoding="utf-8") as f:
    json.dump(classified_data, f, indent=4, ensure_ascii=False)

print("Dataset classification complete!")
