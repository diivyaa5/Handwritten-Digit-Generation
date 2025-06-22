# MNIST Handwritten Digit Generator Web App

This web application generates realistic images of handwritten digits (0–9) based on the MNIST dataset.  
It uses a Variational Autoencoder (VAE) trained from scratch to create diverse digit images that mimic real handwritten samples.

## Features
- Select any digit from 0 to 9
- Generate 5 unique images of the selected digit
- Interactive and easy-to-use interface powered by Streamlit

## How to Run Locally

1. Clone this repository
2. Install dependencies:
   
   ```bash
   pip install torch torchvision streamlit matplotlib
Train the model or download the pretrained model (vae_mnist.pth)

Run the Streamlit app:

bash
streamlit run app.py

Open http://localhost:8501 in your browser

Demo
Example: Generating handwritten digit "3"

Model Training
The Variational Autoencoder (VAE) model is trained from scratch on the MNIST dataset using PyTorch. The training script vae_train.py handles downloading data, training the model, and saving it.

Deployment
The app can be deployed easily on Streamlit Cloud for public access.
