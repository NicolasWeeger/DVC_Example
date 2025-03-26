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
    
    X = torch.tensor(X).unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
    y = torch.tensor(y)
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Ensure correct embedding size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # Ensure correct dimensionality
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, features)
        x = self.transformer_encoder(x)
        x = x[-1]  # Take the last timestep
        x = self.fc(x)
        return x

def train_transformer(file_path, epochs=10, lr=0.001):
    dataloader = prepare_data(file_path)
    model = TransformerModel(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    from dvclive import Live

    with Live() as live:      
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.squeeze(-1)  # Ensure correct input shape (batch, seq_len, features)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
            
            live.log_metric("loss", loss.item())  # Log loss with DVC
            live.next_step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
        torch.save(model.state_dict(), "models/transformer_model.pth")
        live.log_artifact("models/transformer_model.pth", type = "model")  # Save experiment results
    print("Model training complete and saved as models/transformer_model.pth")

if __name__ == "__main__":
    ticker = "NVDA"
    file_path = f"data/raw/{ticker}.parquet"
    train_transformer(file_path, epochs=10, lr=0.001)
