from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("nbertagnolli/counsel-chat")
print(dataset[:100])

# Convert to DataFrame (we use only the 'question' and 'answer' columns)
df = pd.DataFrame(
    {"question": dataset["train"]["question"], "answer": dataset["train"]["answer"]}
)

# Save to CSV for translation
df.to_csv("counselchat_english.csv", index=False)
print("Dataset saved as 'counselchat_english.csv'")

import deepl
import pandas as pd

# Load the CSV
df = pd.read_csv("counselchat_english.csv")

# Initialize DeepL API (replace with your API key)
translator = deepl.Translator("your-api-key")


# Translate text in batches
def translate_text(text):
    try:
        return translator.translate_text(text, source_lang="EN", target_lang="HE").text
    except Exception as e:
        print(f"Error translating: {e}")
        return text  # Fallback to English if error


df["question_hebrew"] = df["question"].apply(translate_text)
df["answer_hebrew"] = df["answer"].apply(translate_text)

# Save Hebrew dataset
df.to_csv("counselchat_hebrew.csv", index=False)
print("Translated dataset saved as 'counselchat_hebrew.csv'")


