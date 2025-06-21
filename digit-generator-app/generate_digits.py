import os
import torch
import gdown
from cvae_model import CVAE  # Make sure this file defines your CVAE model properly

def load_model(path="cvae.pth", device=None):
    if not os.path.exists(path):
        print("Model not found locally. Downloading from Google Drive...")
        gdown.download(
            "https://drive.google.com/uc?id=1_Ikby1WDQzilnnIXhiyEu0G5Z_viGA2D", 
            path, quiet=False
        )

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

    # ✅ z and labels must be PyTorch tensors
    z = torch.randn(num_images, model.latent_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        samples = model.decode(z, labels).cpu()

    return samples.numpy()  # returns a NumPy array to visualize in Streamlit
