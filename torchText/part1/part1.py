"""
LSTM Text Classification using TorchText

Pipeline:
1. Load dataset (JSON/CSV/TSV)
2. Tokenize text using SpaCy
3. Build vocabulary + load pretrained GloVe embeddings
4. Create padded batches using BucketIterator
5. Train an LSTM network for binary classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, TabularDataset, BucketIterator

# -----------------------------------------------------------
# DEVICE CONFIGURATION
# -----------------------------------------------------------

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# TOKENIZER
# -----------------------------------------------------------

# Load SpaCy English tokenizer
# Run once in terminal if not installed:
# python -m spacy download en_core_web_sm
spacy_en = spacy.load("en_core_web_sm")


def tokenize(text):
    """
    Tokenizes input text into individual tokens (words)

    Example:
        "I love AI"
        -> ["I", "love", "AI"]
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# -----------------------------------------------------------
# DEFINE TORCHTEXT FIELDS
# -----------------------------------------------------------

"""
Field defines how each column in the dataset should be processed.

quote:
    - sequential text
    - build vocabulary
    - tokenize using SpaCy

score:
    - numerical label (no vocabulary required)
"""

quote = Field(
    sequential=True,
    use_vocab=True,
    tokenize=tokenize,
    lower=True
)

score = Field(
    sequential=False,
    use_vocab=False
)

# Map dataset columns -> internal variables
fields = {
    "quote": ("q", quote),
    "score": ("s", score)
}


# -----------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------

"""
Dataset structure:

mydata/
    train.json
    test.json

Example JSON entry:
{
    "quote": "Life is beautiful",
    "score": 1
}
"""

train_data, test_data = TabularDataset.splits(
    path="mydata",
    train="train.json",
    test="test.json",
    format="json",
    fields=fields
)


# -----------------------------------------------------------
# BUILD VOCABULARY
# -----------------------------------------------------------

"""
Build vocabulary from training data.

max_size=10000
    Limit vocabulary size to most frequent 10k words

min_freq=1
    Ignore words appearing less than once

vectors="glove.6B.100d"
    Load pretrained 100-dimensional GloVe embeddings
"""

quote.build_vocab(
    train_data,
    max_size=10000,
    min_freq=1,
    vectors="glove.6B.100d"
)


# -----------------------------------------------------------
# CREATE DATA ITERATORS (BATCHING + PADDING)
# -----------------------------------------------------------

"""
BucketIterator:
    - Creates mini-batches
    - Automatically pads sequences
    - Groups sentences of similar lengths
"""

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    device=device
)


# -----------------------------------------------------------
# LSTM MODEL
# -----------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    LSTM based text classifier.

    Architecture:
        Input tokens
            ↓
        Embedding Layer
            ↓
        LSTM
            ↓
        Fully Connected Layer
            ↓
        Binary Prediction
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers
        )

        # Final classifier layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass

        Input shape:
            (sequence_length, batch_size)
        """

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)

        # Convert tokens -> embeddings
        embedded = self.embedding(x)

        # Pass embeddings through LSTM
        outputs, _ = self.lstm(embedded, (h0, c0))

        # Use last time-step output for classification
        final_hidden = outputs[-1]

        # Linear layer
        prediction = self.fc(final_hidden)

        return prediction


# -----------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------

vocab_size = len(quote.vocab)
embedding_dim = 100
hidden_size = 512
num_layers = 2
learning_rate = 0.005
num_epochs = 10


# -----------------------------------------------------------
# INITIALIZE MODEL
# -----------------------------------------------------------

model = LSTMClassifier(
    vocab_size,
    embedding_dim,
    hidden_size,
    num_layers
).to(device)


# -----------------------------------------------------------
# LOAD PRETRAINED GLOVE EMBEDDINGS
# -----------------------------------------------------------

"""
Copy pretrained GloVe embeddings into the embedding layer.

This gives the model semantic knowledge before training.
"""

pretrained_embeddings = quote.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)


# -----------------------------------------------------------
# LOSS FUNCTION & OPTIMIZER
# -----------------------------------------------------------

# Binary classification loss
criterion = nn.BCEWithLogitsLoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# -----------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------

for epoch in range(num_epochs):

    for batch_idx, batch in enumerate(train_iterator):

        # Move batch data to device
        data = batch.q.to(device)
        targets = batch.s.to(device)

        # Forward pass
        predictions = model(data)

        # Compute loss
        loss = criterion(
            predictions.squeeze(1),
            targets.type_as(predictions)
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")