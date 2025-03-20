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

--> you are now set to train a model with (locally) versioned data

## Add dvc experiment tracking to model training
```
with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        
        train_model(...)
        metric_name = evaluate_model(...) # your training code here
        
        live.log_metric(metric_name, value)
        live.next_step()
    
    torch.save(model.state_dict(), "models/model.pth")
    live.log_artifact("model.pth", type="model")
```