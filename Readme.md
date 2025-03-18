# Prepare workspace
## Setup workspace
### Setup and activate virtual environment
```
python -m venv .venv && source .venv/bin/activate
```
### Install dependencies
```
pip install -r requirements.txt
```

## Initialize git and dvc repo
```
git init && echo ".venv" > .gitignore ; dvc init
```

## Download historical stock data
```
download_stock_data.py
```

--> you are now set to 