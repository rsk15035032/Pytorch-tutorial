"""
Utility functions for Seq2Seq:
- Sentence Translation
- BLEU Score Evaluation
- Model Checkpointing
"""

import torch
import spacy
from torchtext.data.metrics import bleu_score

# ==========================================================
# LOAD TOKENIZER ONCE (IMPORTANT OPTIMIZATION)
# ==========================================================
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


# ==========================================================
# TRANSLATE SENTENCE
# ==========================================================
def translate_sentence(model, sentence, german, english, device, max_length=50):
    """
    Translates a German sentence into English using trained Seq2Seq model
    """

    model.eval()

    # ------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add special tokens
    tokens = [german.init_token] + tokens + [german.eos_token]

    # Convert tokens → indices
    indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to tensor (seq_len, batch_size=1)
    sentence_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)

    # ------------------------------------------------------
    # ENCODER FORWARD PASS
    # ------------------------------------------------------
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    # First input to decoder = <SOS>
    outputs = [english.vocab.stoi["<sos>"]]

    # ------------------------------------------------------
    # DECODER LOOP (word-by-word generation)
    # ------------------------------------------------------
    for _ in range(max_length):
        prev_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(prev_word, hidden, cell)

        best_guess = output.argmax(1).item()
        outputs.append(best_guess)

        # Stop if <EOS> predicted
        if best_guess == english.vocab.stoi["<eos>"]:
            break

    # Convert indices → words
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]  # remove <SOS>


# ==========================================================
# BLEU SCORE EVALUATION
# ==========================================================
def bleu(data, model, german, english, device):
    """
    Computes BLEU score on dataset
    """

    targets = []
    outputs = []

    model.eval()

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        # Get prediction
        prediction = translate_sentence(model, src, german, english, device)

        # Remove <EOS>
        prediction = prediction[:-1]

        outputs.append(prediction)
        targets.append([trg])  # format: list of references

    return bleu_score(outputs, targets)


# ==========================================================
# CHECKPOINT FUNCTIONS
# ==========================================================
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves model + optimizer state
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads model + optimizer state
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])