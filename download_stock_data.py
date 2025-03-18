import os
import yfinance as yf
import pandas as pd
import subprocess

def download(ticker: str, start_date: str = "2015-01-01", end_date: str = "2025-01-01", save_path: str = "data/raw/"):
    """
    Downloads historical stock price data and saves it as a CSV file.
    Then, it is versioned using DVC.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Fetch data from Yahoo Finance
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    if data.empty:
        print("Error: No data received. Check the ticker symbol!")
        return
    
    # Save file as Parquet
    file_path = os.path.join(save_path, f"{ticker}.parquet")
    data.to_parquet(file_path, engine='pyarrow')
    print(f"Data saved at: {file_path}")
    
    # Version data with DVC
    subprocess.run(["dvc", "add", file_path])
    subprocess.run(["git", "add", f"{file_path}.dvc", ".gitignore"])
    subprocess.run(["git", "commit", "-m", f"Add stock data for {ticker} to DVC"])
    
    print(f"{ticker} data successfully versioned with DVC!")


if __name__ == "__main__":
    ticker = "NVDA"  
    download(ticker)