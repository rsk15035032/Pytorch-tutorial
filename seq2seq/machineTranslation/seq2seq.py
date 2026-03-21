"""
Seq2Seq (LSTM) for German → English Translation
- CPU friendly (works without GPU)
- GPU ready (auto uses CUDA if available)
- Clean structure + proper comments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# ==========================================================
# DEVICE SETUP (CPU/GPU)
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# TOKENIZERS (SpaCy)
# ==========================================================
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# ==========================================================
# FIELDS (Preprocessing setup)
# ==========================================================
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

# ==========================================================
# LOAD DATASET (Multi30k)
# ==========================================================
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english),
    root="data"
)

# Build vocabulary
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# ==========================================================
# MODEL COMPONENTS
# ==========================================================

class Encoder(nn.Module):
    """
    Encoder:
    Converts input sequence into hidden + cell states
    """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch_size)
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """
    Decoder:
    Generates output sequence step-by-step
    """
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x: (batch_size) → convert to (1, batch_size)
        x = x.unsqueeze(0)

        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        predictions = self.fc(outputs.squeeze(0))  # (batch_size, vocab_size)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    """
    Full Seq2Seq Model:
    Combines Encoder + Decoder with Teacher Forcing
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, vocab_size).to(device)

        # Encode input
        hidden, cell = self.encoder(source)

        # First input to decoder is <SOS>
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            best_guess = output.argmax(1)

            # Teacher forcing
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# ==========================================================
# HYPERPARAMETERS
# ==========================================================
num_epochs = 20            # Reduced for CPU friendliness
learning_rate = 3e-4
batch_size = 32            # Smaller batch for CPU

embedding_size = 256       # Reduced for efficiency
hidden_size = 512
num_layers = 2
dropout = 0.5

# ==========================================================
# DATA ITERATORS
# ==========================================================
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# ==========================================================
# MODEL INITIALIZATION
# ==========================================================
encoder = Encoder(len(german.vocab), embedding_size, hidden_size, num_layers, dropout).to(device)
decoder = Decoder(len(english.vocab), embedding_size, hidden_size, len(english.vocab), num_layers, dropout).to(device)

model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# TensorBoard
writer = SummaryWriter("runs/seq2seq")
step = 0

# ==========================================================
# TRAINING LOOP
# ==========================================================
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(train_iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward pass
        output = model(src, trg)

        # Reshape for loss
        output = output[1:].reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)

        # Backprop
        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping (important for RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        epoch_loss += loss.item()

        writer.add_scalar("Loss", loss.item(), step)
        step += 1

    print(f"Loss: {epoch_loss/len(train_iterator):.4f}")

    # ======================================================
    # EVALUATION (Sample Translation)
    # ======================================================
    model.eval()
    test_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    translated = translate_sentence(model, test_sentence, german, english, device)
    print("Sample Translation:", translated) 

# ==========================================================
# BLEU SCORE EVALUATION
# ==========================================================
score = bleu(test_data[:100], model, german, english, device)
print(f"\nBLEU Score: {score*100:.2f}")