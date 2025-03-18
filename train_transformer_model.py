import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def prepare_data(file_path):
    data = pd.read_parquet(file_path)
    close_prices = data['Close'].values.astype(np.float32)
    seq_length = 30
    X, y = [], []
    for i in range(len(close_prices) - seq_length):
        X.append(close_prices[i:i+seq_length])
        y.append(close_prices[i+seq_length])
    
    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y)
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

def train_transformer(file_path):
    dataloader = prepare_data(file_path)
    model = TransformerModel(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model training complete and saved as transformer_model.pth")

if __name__ == "__main__":
    ticker = "NVDA"
    file_path = f"data/raw/{ticker}.parquet"
    train_transformer(file_path)
