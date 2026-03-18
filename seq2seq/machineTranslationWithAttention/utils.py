import torch
import spacy
from torchtext.data.metrics import bleu_score

# =========================
# Load tokenizer once (IMPORTANT for speed)
# =========================
spacy_ger = spacy.load("de_core_news_sm")


# =========================
# Translate Sentence
# =========================
def translate_sentence(model, sentence, german, english, device, max_length=50):
    """
    Translates a German sentence into English using trained Seq2Seq model.
    """

    model.eval()

    # Tokenize input sentence
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add special tokens
    tokens = [german.init_token] + tokens + [german.eos_token]

    # Convert tokens -> indices
    indices = [german.vocab.stoi.get(token, german.vocab.stoi["<unk>"]) for token in tokens]

    # Convert to tensor (seq_len, batch=1)
    sentence_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        prev_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(
                prev_word, encoder_states, hidden, cell
            )

        best_guess = output.argmax(1).item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    # Convert indices -> tokens
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]  # remove <sos>


# =========================
# BLEU Score Evaluation
# =========================
def bleu(data, model, german, english, device):
    """
    Computes BLEU score on dataset.
    """

    targets = []
    outputs = []

    model.eval()

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)

        # Remove <eos>
        prediction = prediction[:-1]

        outputs.append(prediction)
        targets.append([trg])

    return bleu_score(outputs, targets)


# =========================
# Save Model Checkpoint
# =========================
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# =========================
# Load Model Checkpoint
# =========================
def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")

    model.load_state_dict(checkpoint["state_dict"])

    # Optimizer is optional (useful for inference-only loading)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])