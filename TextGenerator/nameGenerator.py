"""
Character Level RNN Text Generator
----------------------------------

This model learns character sequences from a text file
and generates new text using an LSTM.

Example Use Case:
- Name generation
- Story generation
- Code generation
- Character modeling

The model works on both CPU and GPU.
"""

import torch
import torch.nn as nn
import random
import string
import unidecode
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For GPU performance (commented for CPU-only environments)
# torch.backends.cudnn.benchmark = True


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

# Use all printable characters
all_characters = string.printable
n_characters = len(all_characters)


# ============================================================
# LOAD DATASET
# ============================================================

# Any large text file can be used
# Example: names.txt, books, poems, code, etc.

file = unidecode.unidecode(open("data/names.txt").read())


# ============================================================
# CHARACTER LEVEL RNN MODEL
# ============================================================

class CharRNN(nn.Module):
    """
    Character-level LSTM network
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CharRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convert characters → dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM network
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):

        # Character → embedding
        x = self.embedding(x)

        # LSTM forward
        out, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))

        # Flatten output
        out = self.fc(out.reshape(out.shape[0], -1))

        return out, (hidden, cell)

    def init_hidden(self, batch_size):

        # Initialize hidden and cell states

        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size
        ).to(device)

        cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size
        ).to(device)

        return hidden, cell


# ============================================================
# TEXT GENERATOR CLASS
# ============================================================

class TextGenerator:

    def __init__(self):

        # ====================================================
        # TRAINING HYPERPARAMETERS
        # ====================================================

        self.chunk_len = 200       # shorter for CPU efficiency
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 100

        # Model parameters
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003

    # --------------------------------------------------------
    # Convert string → tensor of character indices
    # --------------------------------------------------------

    def char_tensor(self, text):

        tensor = torch.zeros(len(text)).long()

        for i in range(len(text)):
            tensor[i] = all_characters.index(text[i])

        return tensor

    # --------------------------------------------------------
    # Create random training batch
    # --------------------------------------------------------

    def get_random_batch(self):

        start_index = random.randint(0, len(file) - self.chunk_len - 1)

        end_index = start_index + self.chunk_len + 1

        text = file[start_index:end_index]

        inp = torch.zeros(self.batch_size, self.chunk_len)
        target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):

            inp[i] = self.char_tensor(text[:-1])
            target[i] = self.char_tensor(text[1:])

        return inp.long(), target.long()

    # --------------------------------------------------------
    # TEXT GENERATION (INFERENCE)
    # --------------------------------------------------------

    def generate(self, start_string="A", predict_len=120, temperature=0.8):

        hidden, cell = self.model.init_hidden(self.batch_size)

        input_tensor = self.char_tensor(start_string)

        predicted = start_string

        # Warm up LSTM with starting string
        for i in range(len(start_string) - 1):

            _, (hidden, cell) = self.model(
                input_tensor[i].view(1).to(device),
                hidden,
                cell
            )

        last_char = input_tensor[-1]

        # Generate characters
        for _ in range(predict_len):

            output, (hidden, cell) = self.model(
                last_char.view(1).to(device),
                hidden,
                cell
            )

            # Temperature sampling
            output_dist = output.data.view(-1).div(temperature).exp()

            top_char = torch.multinomial(output_dist, 1)[0]

            predicted_char = all_characters[top_char]

            predicted += predicted_char

            last_char = self.char_tensor(predicted_char)

        return predicted

    # --------------------------------------------------------
    # TRAINING FUNCTION
    # --------------------------------------------------------

    def train(self):

        # Create model
        self.model = CharRNN(
            n_characters,
            self.hidden_size,
            self.num_layers,
            n_characters
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # TensorBoard logging
        writer = SummaryWriter("runs/char_rnn")

        print("🚀 Starting Training...\n")

        # Progress bar over epochs
        progress_bar = tqdm(range(1, self.num_epochs + 1), desc="Training")

        for epoch in progress_bar:

            inp, target = self.get_random_batch()

            inp = inp.to(device)
            target = target.to(device)

            hidden, cell = self.model.init_hidden(self.batch_size)

            self.model.zero_grad()

            loss = 0

            # Process sequence character by character
            for c in range(self.chunk_len):

                output, (hidden, cell) = self.model(
                    inp[:, c],
                    hidden,
                    cell
                )

                loss += criterion(output, target[:, c])

            # Backpropagation
            loss.backward()

            optimizer.step()

            loss = loss.item() / self.chunk_len

            
           # Update progress bar with loss
            progress_bar.set_postfix(loss=loss)

            # Print generated sample occasionally
            if epoch % self.print_every == 0:

                print("\nGenerated Text:\n")
                print(self.generate())
                print("-" * 60)

        writer.add_scalar("Training Loss", loss, epoch)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    generator = TextGenerator()

    generator.train()