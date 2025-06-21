import os
import torch
import gdown
from cvae_model import CVAE  # make sure this exists and defines your model correctly

def load_model(path="cvae.pth", device=None):
    if not os.path.exists(path):
        print("Model not found locally. Downloading...")
        gdown.download(
            "https://drive.google.com/uc?id=1_Ikby1WDQzilnnIXhiyEu0G5Z_viGA2D",
            path,
            quiet=False
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

    # ✅ Generate random latent vectors using torch (NOT numpy)
    z = torch.randn(num_images, model.latent_dim, device=device)

    # ✅ Generate digit labels as torch tensor
    labels = torch.full((num_images,), digit, dtype=torch.long, device=device)

    # ✅ Model expects tensors — decode then detach and convert to numpy for visualization
    with torch.no_grad():
        outputs = model.decode(z, labels).cpu()

    # ✅ Return as numpy array for matplotlib display
    return outputs.numpy()
