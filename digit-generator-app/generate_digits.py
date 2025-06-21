import os
import torch
import gdown
from cvae_model import CVAE
import numpy as np

def load_model(path="cvae.pth", device=None):
    if not os.path.exists(path):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1_Ikby1WDQzilnnIXhiyEu0G5Z_viGA2D"
        gdown.download(url, path, quiet=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_images(model, digit, num_images=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = torch.randn(num_images, model.latent_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        samples = model.decode(z, labels).cpu().numpy()
    return samples
