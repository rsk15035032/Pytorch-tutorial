import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 356

# Image loader
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Load images
original_img = load_image("nuralStyle/TaraSutaria.jpg")
style_img = load_image("nuralStyle/style.jpg")

# VGG Model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(
            weights=VGG19_Weights.IMAGENET1K_V1
        ).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def gram_matrix(x):
    b, c, h, w = x.shape
    features = x.view(c, h * w)
    G = features @ features.t()
    return G / (c * h * w)

# Initialize model
model = VGG().to(device).eval()

# Freeze VGG
for param in model.parameters():
    param.requires_grad = False

# Precompute features (IMPORTANT)
original_features = model(original_img)
style_features = model(style_img)

style_grams = [gram_matrix(f) for f in style_features]

# Generated image
generated = original_img.clone().requires_grad_(True)

# Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 1e4

optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):

    gen_features = model(generated)

    content_loss = 0
    style_loss = 0

    for gen_feat, orig_feat, style_gram in zip(
        gen_features, original_features, style_grams
    ):
        content_loss += torch.mean((gen_feat - orig_feat) ** 2)

        G = gram_matrix(gen_feat)
        style_loss += torch.mean((G - style_gram) ** 2)

    total_loss = alpha * content_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Clamp image values
    generated.data.clamp_(0, 1)

    if step % 200 == 0:
        print(f"Step {step} | Loss: {total_loss.item()}")
        save_image(generated, "nuralStyle/generated.png")