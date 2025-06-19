import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)

        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.permute(0, 2, 1)
        return trend_vals


class LevelModel(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len):
        super(LevelModel, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.level_dense1 = nn.Linear(self.latent_dim, self.feat_dim)
        self.level_dense2 = nn.Linear(self.feat_dim, self.feat_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.feat_dim)

        ones_tensor = torch.ones((1, self.seq_len, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals

class ResidualConnection(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim):
        super(ResidualConnection, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # Fully connected layer to project latent space to residuals
        self.fc = nn.Linear(latent_dim, seq_len * feat_dim)

    def forward(self, z):
        # Project latent space to residuals
        residuals = self.fc(z)
        # Reshape to match the output dimensions
        return residuals.view(-1, self.seq_len, self.feat_dim)


class TrendVAEEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super(TrendVAEEncoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes

        layers = []
        layers.append(nn.Conv1d(feat_dim, hidden_layer_sizes[0], kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_layer_sizes)):
            layers.append(nn.Conv1d(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_layer_sizes)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, feat_dim, seq_len)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_layer_sizes):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for layer in self.encoder:
                x = layer(x)
            return x.numel()


class TrendVAEDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim, trend_poly=0, use_residual_conn=True):
        super(TrendVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.use_residual_conn = use_residual_conn

        # Level and Trend Layers
        self.level_model = LevelModel(latent_dim, feat_dim, seq_len)
        self.trend_layer = TrendLayer(seq_len, feat_dim, latent_dim, trend_poly)

        # Residual Connection (optional)
        if use_residual_conn:
            self.residual_conn = ResidualConnection(seq_len, feat_dim, latent_dim)

        # Convolutional Decoder
        layers = []
        layers.append(nn.Conv1d(latent_dim, hidden_layer_sizes[-1], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layer_sizes) - 1, 0, -1):
            layers.append(nn.Conv1d(hidden_layer_sizes[i], hidden_layer_sizes[i - 1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(hidden_layer_sizes[0], feat_dim, kernel_size=3, stride=1, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        # Compute level and trend components
        level_vals = self.level_model(z)  # Shape: (batch_size, seq_len, feat_dim)
        trend_vals = self.trend_layer(z)  # Shape: (batch_size, seq_len, feat_dim)

        # Compute residuals (if enabled)
        residuals = 0
        if self.use_residual_conn:
            residuals = self.residual_conn(z)  # Shape: (batch_size, seq_len, feat_dim)

        # Pass latent variable through convolutional decoder
        # Z shape: (batch_size, latent_dim)
        # z = z.unsqueeze(2).repeat(1, 1, self.seq_len)  # Reshape to (batch_size, latent_dim, seq_len)
        z = z.unsqueeze(2).expand(-1, -1, self.seq_len)  # Expand to (batch_size, latent_dim, seq_len)
        x_hat = self.decoder(z)  # Shape: (batch_size, feat_dim, seq_len)
        x_hat = x_hat.permute(0, 2, 1)  # Change to (batch_size, seq_len, feat_dim)

        return x_hat + level_vals + trend_vals + residuals


class Trend_Vae(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dim, latent_dim, z_dim, trend_poly=0, use_residual_conn=True, device='cpu'):
        super(Trend_Vae, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.trend_poly = trend_poly
        self.use_residual_conn = use_residual_conn

        # Encoder
        self.encoder = TrendVAEEncoder(seq_len, feat_dim, [hidden_dim, latent_dim], z_dim)

        # Decoder
        self.decoder = TrendVAEDecoder(
            seq_len=seq_len,
            feat_dim=feat_dim,
            hidden_layer_sizes=[latent_dim, hidden_dim],
            latent_dim=z_dim,
            trend_poly=trend_poly,
            use_residual_conn=use_residual_conn
        )

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(self.device)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        return z

    def forward(self, x):
        # Encode input to latent space
        mean, log_var = self.encoder(x)

        # Reparameterize to sample latent variable
        z = self.reparameterization(mean, log_var)

        # Decode latent variable to reconstruct input
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
    
    def generate(self, num_samples):
        """
        Generate new data using random latent vectors.
        Args:
            num_samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated data of shape (num_samples, seq_len, feat_dim).
        """
        # Generate random latent vectors
        latent_vectors = torch.randn((num_samples, self.z_dim)).to(self.device)

        # Decode latent vectors to generate data
        with torch.no_grad():
            generated_data = self.decoder(latent_vectors)

        return generated_data


def vae_loss_function(x, x_hat, mean, var, beta=1):
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + torch.log(var.pow(2)) - mean.pow(2) - var.exp())
    return reconstruction_loss + beta * KLD


class ClimateDataset(Dataset):
    def __init__(self, data):
        print("Creating datasets...")
        self.inputs = []
        self.outputs = []
        for model, runs in tqdm(data.items(), desc="Processing models"):
            forced_response = runs['forced_response']
            for run_key, run_data in runs.items():
                if run_key != 'forced_response':
                    self.inputs.append(run_data)
                    self.outputs.append(forced_response)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.outputs = torch.tensor(np.array(self.outputs), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]
        return {'input': input_data, 'output': output_data}


def train_vae(model, data_loader, optimizer, epochs, device='cpu'):
    model.to(device)
    model.train()
    losses = []

    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch in data_loader:
            x = batch[0].to(device)  # Shape: (N, T, D)
            y = batch[1].to(device)  # Shape: (N, T, D)

            x_hat, mean, var = model(x)
            loss = vae_loss_function(y, x_hat, mean, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        avg_loss = overall_loss / len(data_loader.dataset)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return losses

def save_model(model, path):
    # If the directory does not exist, create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))

def generate_confidence_interval(model, input_tensor, num_samples=100, confidence=0.95):
    """
    Generate multiple predictions using the decoder of the Trend_Vae model
    and calculate the confidence interval for the output.

    Args:
        model (Trend_Vae): The trained Trend_Vae model.
        input_tensor (torch.Tensor): Input tensor to encode and decode.
        num_samples (int): Number of samples to generate.
        confidence (float): Confidence level for the interval (e.g., 0.95).

    Returns:
        tuple: Mean prediction, lower bound, and upper bound of the confidence interval.
    """
    model.eval()
    with torch.no_grad():
        # Encode the input to latent space
        mean, log_var = model.encoder(input_tensor)
        
        # Generate multiple samples from the latent space
        samples = []
        for _ in range(num_samples):
            z = model.reparameterization(mean, log_var)
            decoded_output = model.decoder(z)
            samples.append(decoded_output.cpu().numpy())

        # Convert samples to a numpy array for statistical calculations
        samples = np.array(samples)  # Shape: (num_samples, batch_size, seq_len, feat_dim)

        # Calculate mean and confidence intervals
        mean_prediction = np.mean(samples, axis=0)
        lower_bound = np.percentile(samples, (1 - confidence) / 2 * 100, axis=0)
        upper_bound = np.percentile(samples, (1 + confidence) / 2 * 100, axis=0)
    return mean_prediction, lower_bound, upper_bound