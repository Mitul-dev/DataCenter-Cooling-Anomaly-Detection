import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, features, width=24, hidden=64, latent=16):
        super().__init__()
        self.width = width
        self.encoder = nn.LSTM(features, hidden, batch_first=True)
        self.bottleneck = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.output = nn.Linear(hidden, features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = self.bottleneck(h[-1])
        decoded, _ = self.decoder(latent[:, None, :].expand(-1, self.width, -1).contiguous())
        return self.output(decoded)


def predict_errors(model, x, batch_size=1024, device="cpu"):
    model.eval()
    mean, endpoint, feature = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            inputs = torch.from_numpy(x[start:start+batch_size]).to(device)
            error = (inputs - model(inputs)).square().cpu().numpy()
            mean.append(error.mean(axis=(1,2)))
            endpoint.append(error[:, -1].mean(axis=1))
            feature.append(error[:, -1])
    return np.concatenate(mean), np.concatenate(endpoint), np.concatenate(feature)
