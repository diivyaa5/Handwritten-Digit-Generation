# app.py
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

# Load the model
device = torch.device("cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

# UI
st.title("MNIST Handwritten Digit Generator")
digit = st.selectbox("Select a digit (just for labeling purposes):", list(range(10)))

st.write(f"Generating 5 samples of handwritten '{digit}'...")

cols = st.columns(5)
for i in range(5):
    z = torch.randn(1, 20).to(device)
    with torch.no_grad():
        img = model.decode(z).view(28, 28).numpy()

    cols[i].image(img, width=100, clamp=True, caption=f"{digit}")
