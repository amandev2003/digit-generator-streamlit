# Handwritten Digit Generator (0–9) using CVAE + Streamlit

This is a web application that generates handwritten digits (0–9) using a **Conditional Variational Autoencoder (CVAE)** trained on the **MNIST dataset**. The app is built with **Streamlit** and hosted on **Streamlit Cloud**.

---

## ✨ Features

- Select any digit from **0 to 9**
- Generate **5 diverse images** of that digit (no repetitions)
- Images resemble the style and format of **MNIST digits**
- Powered by a **deep generative model trained from scratch** (no pre-trained weights)

---

## 📦 Files in This Repository

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app UI |
| `generate_digits.py` | Loads trained model and generates digit images |
| `cvae_model.py` | Contains the Conditional VAE model class |
| `cvae.pth` | Trained PyTorch model weights |
| `requirements.txt` | List of required Python packages |
| `README.md` | This file |

---

## 🧠 Model Architecture

The model is a **Conditional Variational Autoencoder (CVAE)** implemented in **PyTorch**:

- **Input:** 28×28 grayscale image + class label (0–9)
- **Latent vector:** 20 dimensions
- **Encoder:** Fully connected
- **Decoder:** Fully connected with conditional label embedding

The model was trained on **Google Colab** using the **T4 GPU** with the MNIST dataset.

---

## 🛠 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/digit-generator-streamlit.git
cd digit-generator-streamlit
