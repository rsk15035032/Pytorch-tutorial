"""
TorchText Translation Data Pipeline (CPU Friendly + GPU Ready)

This script:
1. Loads parallel English–German sentences from text files
2. Splits them into train and test sets
3. Saves them into JSON/CSV formats
4. Uses torchtext TabularDataset to load the data
5. Tokenizes using spaCy
6. Builds vocabulary
7. Creates batch iterators for training

Works on both CPU and GPU automatically.
"""

# ==========================================================
# Imports
# ==========================================================
import torch
import spacy
import pandas as pd

from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split


# ==========================================================
# Device Configuration (CPU Friendly + GPU Ready)
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================================
# Load Raw Text Data
# Each line in file represents one sentence
# ==========================================================
english_txt = open("torchText/part3/train_WMT_english.txt", encoding="utf8").read().split("\n")
german_txt = open("torchText/part3/train_WMT_german.txt", encoding="utf8").read().split("\n")

# Create dictionary of paired sentences
raw_data = {
    "English": [line for line in english_txt[1:100]],
    "German": [line for line in german_txt[1:100]],
}

# Convert to pandas DataFrame
df = pd.DataFrame(raw_data, columns=["English", "German"])


# ==========================================================
# Train/Test Split
# ==========================================================
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Save dataset in JSON format (recommended for torchtext)
train_df.to_json("train.json", orient="records", lines=True)
test_df.to_json("test.json", orient="records", lines=True)

# Also save CSV version (optional)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)


# ==========================================================
# Load spaCy Tokenizers
# Install using:
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
# ==========================================================
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


# ==========================================================
# Tokenization Functions
# Convert sentence → list of tokens
# ==========================================================
def tokenize_eng(text):
    return [token.text for token in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]


# ==========================================================
# Define TorchText Fields
# These control how text is processed
# ==========================================================
english = Field(
    sequential=True,
    use_vocab=True,
    tokenize=tokenize_eng,
    lower=True
)

german = Field(
    sequential=True,
    use_vocab=True,
    tokenize=tokenize_ger,
    lower=True
)


# Map dataset columns → torchtext fields
fields = {
    "English": ("eng", english),
    "German": ("ger", german),
}


# ==========================================================
# Load Dataset using TorchText TabularDataset
# ==========================================================
train_data, test_data = TabularDataset.splits(
    path="",
    train="train.json",
    test="test.json",
    format="json",
    fields=fields
)


# ==========================================================
# Build Vocabulary
# Only words appearing >=2 times are kept
# ==========================================================
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

print(f"English vocab size: {len(english.vocab)}")
print(f"German vocab size: {len(german.vocab)}")


# ==========================================================
# Create Iterators (Batches)
# BucketIterator groups similar length sentences together
# for faster training
# ==========================================================
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=32,
    device=device
)


# ==========================================================
# Example Batch
# ==========================================================
for batch in train_iterator:
    print("English batch shape:", batch.eng.shape)
    print("German batch shape:", batch.ger.shape)
    break


