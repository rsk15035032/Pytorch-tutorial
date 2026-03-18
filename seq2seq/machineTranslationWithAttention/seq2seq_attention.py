import random
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from tqdm import tqdm

from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# =========================
# Load Spacy Tokenizers
# =========================
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# =========================
# Fields & Dataset
# =========================
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english),
    root="data"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.rnn = nn.LSTM(
            emb_size, hidden_size, num_layers, bidirectional=True
        )
        self.embedding = nn.Embedding(input_size, emb_size)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, batch)
        embedding = self.dropout(self.embedding(x))

        encoder_states, (hidden, cell) = self.rnn(embedding)

        # Convert bidirectional -> unidirectional
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


# =========================
# Decoder with Attention
# =========================
class Decoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(hidden_size * 2 + emb_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)  # (1, batch)
        embedding = self.dropout(self.embedding(x))

        seq_len = encoder_states.shape[0]

        # Repeat hidden state for attention
        hidden_repeat = hidden.repeat(seq_len, 1, 1)

        energy = self.relu(
            self.energy(torch.cat((hidden_repeat, encoder_states), dim=2))
        )

        attention = self.softmax(energy)

        # Context vector
        context = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context, embedding), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        predictions = self.fc(outputs).squeeze(0)

        return predictions, hidden, cell


# =========================
# Seq2Seq Wrapper
# =========================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, vocab_size).to(source.device)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]  # <sos>

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# =========================
# Training Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20
learning_rate = 3e-4
batch_size = 32

encoder = Encoder(len(german.vocab), 300, 512, 1, 0.3).to(device)
decoder = Decoder(len(english.vocab), 300, 512, len(english.vocab), 1, 0.3).to(device)

model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

writer = SummaryWriter("runs/loss_plot")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)


# =========================
# Training Loop
# =========================
step = 0

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    model.train()
    progress_bar = tqdm(train_iterator, desc="Training", leave=False)

    for batch in progress_bar:
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)

        # Reshape for loss
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        # Gradient clipping (important for RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

        writer.add_scalar("Training Loss", loss.item(), step)
        step += 1

    # =========================
    # Validation Example
    # =========================
    model.eval()

    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferd gezogen."

    translated = translate_sentence(model, sentence, german, english, device)

    print("Example Translation:", translated)

    # Save checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)


# =========================
# BLEU Score
# =========================
score = bleu(test_data[1:100], model, german, english, device)
print(f"\nBLEU Score: {score * 100:.2f}")