import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from cvae_model import CVAE  # import from your model file if needed

# Load the model
def load_model(path="cvae.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Generate images for a specific digit
def generate_images(model, digit, num_images=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    digit_tensor = torch.tensor([digit] * num_images, dtype=torch.long).to(device)
    z = torch.randn(num_images, model.latent_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z, digit_tensor)
    return generated.cpu()
