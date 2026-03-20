import torch
import spacy
from torchtext.data.metrics import bleu_score


# =====================================================
# Load tokenizer only once (much faster than loading
# inside translate_sentence every time)
# =====================================================
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

# =====================================================
# Translate a single sentence using trained Transformer
# =====================================================
def translate_sentence(model, sentence, german, english, device, max_length=50):
    model.eval()

    # If input is a string → tokenize using spacy
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <sos> and <eos> tokens
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Convert tokens → indices
    indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to tensor (shape: seq_len x 1)
    sentence_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)

    # First decoder input = <sos>
    outputs = [english.vocab.stoi["<sos>"]]

    # Generate tokens one by one
    for _ in range(max_length):

        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        # No gradient needed during inference
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        # Get best predicted token
        best_token = output.argmax(2)[-1, :].item()
        outputs.append(best_token)

        # Stop when <eos> is generated
        if best_token == english.vocab.stoi["<eos>"]:
            break

    # Convert indices → tokens
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # Remove <sos>
    return translated_sentence[1:]


# =====================================================
# Compute BLEU score on dataset
# =====================================================
def bleu(data, model, german, english, device):

    model.eval()

    targets = []
    outputs = []

    for example in data:

        src = vars(example)["src"]
        trg = vars(example)["trg"]

        # Predict translation
        prediction = translate_sentence(model, src, german, english, device)

        # Remove <eos>
        prediction = prediction[:-1]

        outputs.append(prediction)
        targets.append([trg])

    return bleu_score(outputs, targets)


# =====================================================
# Save model checkpoint
# =====================================================
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# =====================================================
# Load model checkpoint
# =====================================================
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])