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

## Download or update historical stock data until today
```
update_stock_data.py
```

--> you are now set to 