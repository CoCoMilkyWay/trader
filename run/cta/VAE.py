import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, lr=1e-3, batch_size=64, epochs=50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Optimizer (will be set after model is moved to device)
        self.optimizer = None

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train_model(self, data_array):
        """
        Train the VAE on numpy or pandas array data (shape [N, D]).
        """
        # Convert to tensor and dataset
        X = torch.tensor(data_array, dtype=torch.float32)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.forward(x_batch)
                loss = self.loss_function(recon, x_batch, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss:.2f}")

    def encode_data(self, data_array):
        """
        Encode data to latent space means.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data_array, dtype=torch.float32).to(self.device)
            mu, _ = self.encode(x)
        return mu.cpu().numpy()

    def reconstruct(self, data_array):
        """
        Reconstruct input data via the autoencoder.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data_array, dtype=torch.float32).to(self.device)
            recon, _, _ = self.forward(x)
        return recon.cpu().numpy()
# 
# import pandas as pd
# import numpy as np
# 
# # Example dataframe (replace with your real data)
# df = pd.DataFrame(np.random.randn(1000, 20))
# X = df.values.astype(np.float32)
# 
# # Optional normalization
# X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
# 
# vae = VAE(input_dim=X.shape[1], hidden_dim=64, latent_dim=8, epochs=30)
# vae.train_model(X)
# 
# # Get latent representation means
# latent_means = vae.encode_data(X)
# 
# # Reconstruct inputs
# reconstructed = vae.reconstruct(X)
# 