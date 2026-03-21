"""
FINAL CLEAN VERSION
Transformer Seq2Seq model on Multi30k (German -> English)

Features:
- CPU friendly
- GPU ready
- tqdm progress bar
- TensorBoard logging (runs/ folder always created)
- Checkpoints always saved in project folder
- Translation example every epoch
- BLEU score at the end
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
# Create required folders
# =========================
BASE_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE_DIR, "runs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# Load spacy tokenizers
# =========================
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# =========================
# Fields
# =========================
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")


# =========================
# Load dataset
# =========================
DATA_PATH = os.path.join(BASE_DIR, "data")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english),
    root=DATA_PATH
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# =========================
# Transformer Model
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
        super().__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        self.src_position_embedding = nn.Embedding(max_len, embed_size)
        self.trg_position_embedding = nn.Embedding(max_len, embed_size)

        self.transformer = nn.Transformer(
            embed_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        return (src.transpose(0, 1) == self.src_pad_idx)

    def forward(self, src, trg):

        src_len, batch_size = src.shape
        trg_len, batch_size = trg.shape

        src_positions = (
            torch.arange(0, src_len)
            .unsqueeze(1)
            .expand(src_len, batch_size)
            .to(device)
        )

        trg_positions = (
            torch.arange(0, trg_len)
            .unsqueeze(1)
            .expand(trg_len, batch_size)
            .to(device)
        )

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )

        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        src_padding_mask = self.make_src_mask(src).to(device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )

        return self.fc_out(out)


# =========================
# Hyperparameters
# =========================
num_epochs = 5
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
# Data loaders
# =========================
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)


# =========================
# Model / Optimizer / Loss
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

writer = SummaryWriter(RUNS_DIR)
step = 0


# =========================
# Training loop
# =========================
for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0

    loop = tqdm(train_iterator, leave=True)

    for batch_idx, batch in enumerate(loop):

        src = batch.src.to(device)
        trg = batch.trg.to(device)

        output = model(src, trg[:-1, :])

        output = output.reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

        writer.add_scalar("Training Loss", loss.item(), step)
        step += 1

    print(f"Epoch Loss: {epoch_loss/len(train_iterator):.4f}")

    # =========================
    # Save checkpoint every epoch
    # =========================
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print("Checkpoint saved:", checkpoint_path)


       # ======================================================
    # EVALUATION (Sample Translation)
    # ======================================================
    model.eval()
    test_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    translated = translate_sentence(model, test_sentence, german, english, device)
    print("Sample Translation:", translated)

# Close TensorBoard writer
writer.close()


# ==========================================================
# BLEU SCORE EVALUATION
# ==========================================================
score = bleu(test_data[:100], model, german, english, device)
print(f"\nBLEU Score: {score*100:.2f}")





  