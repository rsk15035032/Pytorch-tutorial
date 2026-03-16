# ==========================================================
# TorchText + SpaCy Data Pipeline (CPU/GPU Friendly)
# ==========================================================
# This script loads the Multi30k German-English translation
# dataset, tokenizes text using SpaCy, builds vocabularies,
# and creates efficient batch iterators for training models
# like Seq2Seq, Transformer, or LSTM.
# ==========================================================
import os
import torch
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# ----------------------------------------------------------
# Importing the data folder to avoid the datasets
# ----------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------
# Select device automatically (CPU or GPU)
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------
# Load SpaCy language models
# en_core_web_sm -> English tokenizer + pipeline
# de_core_news_sm -> German tokenizer + pipeline
# ----------------------------------------------------------

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


# ----------------------------------------------------------
# Tokenization functions
# Convert sentence -> list of tokens
# Example: "Hello world!" -> ["Hello", "world", "!"]
# ----------------------------------------------------------
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# ----------------------------------------------------------
# Define Fields (how text should be processed)
#
# sequential=True → data is a sequence (sentence)
# use_vocab=True  → build word-index mapping
# tokenize        → tokenization function
# lower=True      → convert words to lowercase
# ----------------------------------------------------------
english = Field(tokenize=tokenize_eng, lower=True)
german = Field(tokenize=tokenize_ger, lower=True)


# ----------------------------------------------------------
# Load Multi30k dataset
# German (.de) is the source language
# English (.en) is the target language
# ----------------------------------------------------------
train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english),
    root=DATA_PATH   # local dataset folder
)


# ----------------------------------------------------------
# Build vocabulary (word -> integer index)
#
# max_size → maximum vocabulary size
# min_freq → ignore words appearing less than 2 times
# ----------------------------------------------------------
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)


# ----------------------------------------------------------
# Create batch iterators
#
# BucketIterator groups sentences of similar lengths
# together to reduce padding and improve efficiency.
# ----------------------------------------------------------
train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=64,
    device=device,
)


# ----------------------------------------------------------
# Inspect a single batch
# ----------------------------------------------------------
for batch in train_iterator:
    print(batch)
    break


# ----------------------------------------------------------
# Vocabulary lookup examples
# ----------------------------------------------------------

# string -> integer index (stoi = string to index)
print(f'Index of the word "the": {english.vocab.stoi["the"]}')

# integer index -> word (itos = index to string)
print(f"Word at index 1612: {english.vocab.itos[1612]}")
print(f"Word at index 0: {english.vocab.itos[0]}")
