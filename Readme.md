# Prepare workspace
## Setup workspace
### Setup and activate virtual environment
```bash
python -m venv .venv && source .venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Initialize git and DVC repo
```bash
git init && echo ".venv" > .gitignore && dvc init && git commit -m "Initialize git and DVC
```

# Setup
## Add DVC version control to your data
```bash
dvc add "path/to/file.parquet" ; git add . ; git commit -m "Add initial dataset to DVC" # Add your datapath here
```

### Example: Download or update historical stock data until today
```bash
python update_stock_data.py
```

**--> you are now set to train a model with (locally) versioned data**

## Add DVC experiment tracking to model training
```python
with Live() as live:
    live.log_param("epochs", NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        
        train_model(...)
        metric_name = evaluate_model(...) # your training code here
        
        live.log_metric("metric_name", metric_name)
        live.next_step()
    
    torch.save(model.state_dict(), "models/model.pth")
    live.log_artifact("model.pth", type="model")
```
### Example: train transformer model from downloaded stock data
```bash
python train_transformer_model.py
```
**Note: Git commit also handles dvc commit for dvc added data, which is done automatically with the usage of dvclive**

**--> These steps enable you to do manual experiments using the examples `update_stock_data.py` and `train_transformer_model.py`, utilizing DVC to control data versioning and ** 

## Automate experiments using pipelines 