"""
Transformer Seq2Seq model on the Multi30k dataset (German -> English)

Optimized version:
- CPU friendly
- GPU ready
- tqdm progress bar added
- Clean structure
- Proper comments at the right place
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import spacy

from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


# =========================
#  Device configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
#  Load spacy tokenizer
# =========================
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


# ----------------------------------------------------------
# Importing the data folder to avoid the datasets
# ----------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


# Tokenizers
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# =========================
#  Define Fields
# =========================
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")


# =========================
#  Load dataset
# =========================
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english),
    root=DATA_PATH
)

# Build vocabularies
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# =========================
#  Transformer Model
# =========================
class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
    ):
        super(Transformer, self).__init__()

        # Word embeddings
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        # Positional embeddings
        self.src_position_embedding = nn.Embedding(max_len, embed_size)
        self.trg_position_embedding = nn.Embedding(max_len, embed_size)

        # Transformer block (PyTorch built-in)
        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        # Final output layer
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    # Create mask to ignore padding tokens
    def make_src_mask(self, src):
        return (src.transpose(0, 1) == self.src_pad_idx)

    def forward(self, src, trg):

        src_seq_length, batch_size = src.shape
        trg_seq_length, batch_size = trg.shape

        # Create position ids
        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, batch_size)
            .to(device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, batch_size)
            .to(device)
        )

        # Add word embedding + positional embedding
        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )

        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        # Create masks
        src_padding_mask = self.make_src_mask(src).to(device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(device)

        # Transformer forward
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )

        return self.fc_out(out)


# =========================
#  Hyperparameters
# =========================
num_epochs = 20
learning_rate = 3e-4
batch_size = 32

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
forward_expansion = 4
dropout = 0.1
max_len = 100

src_pad_idx = german.vocab.stoi["<pad>"]


# =========================
#  Data loaders
# =========================
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)


# =========================
#  Model, optimizer, loss
# =========================
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

writer = SummaryWriter("runs/transformer_loss")
step = 0


# =========================
#  Training loop with tqdm
# =========================
for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0

    loop = tqdm(train_iterator, leave=True)

    for batch_idx, batch in enumerate(loop):

        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward pass
        output = model(src, trg[:-1, :])

        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping (very important for Transformer stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        epoch_loss += loss.item()

        # Update tqdm bar
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

        writer.add_scalar("Training Loss", loss.item(), global_step=step)
        step += 1

    print(f"Epoch Loss: {epoch_loss/len(train_iterator):.4f}")


# =========================
#  BLEU Score Evaluation
# =========================
score = bleu(test_data[1:100], model, german, english, device)
print(f"BLEU score: {score*100:.2f}")


# ======================
# Translation Example
# ======================
model.eval()

example_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

translated_sentence = translate_sentence(
    model,
    example_sentence,
    german,
    english,
    device,
    max_length=50,
    )

print("\nExample translation:")
print("German :", example_sentence)
print("English:", translated_sentence)

# Save checkpoint
checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
save_checkpoint(checkpoint)



  