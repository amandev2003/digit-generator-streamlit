import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from generate_digits import load_model, generate_images  # Your functions

st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate (0–9)", list(range(10)))

if st.button("Generate"):
    model = load_model()
    images = generate_images(model, digit, num_images=5)

    fig, ax = plt.subplots(figsize=(10, 2))
    grid = make_grid(images, nrow=5, padding=2)
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
