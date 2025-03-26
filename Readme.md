# DVC for data versioning, experiment tracking and model versioning
This is a reference application for using DVC for data versioning, experiment tracking and model versioning for **small to medium datasets**. The scope of this project is to introduce the usage of DVC, show how to use DVC for your own project and give an example implementation for comparison and testing. 

For more information see the official DVC documentation: https://dvc.org/doc/start.

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

# Setup DVC for data and modelversioning and experiment tracking
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
### Example: Train transformer model from downloaded stock data (note: this is just an example and not a good performing model)
```bash
python train_transformer_model.py
```
**Note: Git commit also handles dvc commit for dvc added data, which is done automatically with the usage of dvclive**

**--> These steps enable you to do manual experiments using the examples `update_stock_data.py` and `train_transformer_model.py`, utilizing DVC to control data versioning, experiment tracking and mdoel versioning**

### Vizualize tracked experiments
#### DVC extension for VSCode
- Download the DVC extension for VSCode
- With this extension, you can see and compare the DVC Experiments, show plots, restore models, datasets and parameters to previous experiments
#### Other methods (eg. DVC CLI usage)
- see: https://dvc.org/doc/user-guide/experiment-management/comparing-experiments
 
---

**Note: DVC stores the models in the DVC cache. This can also be used with an external bucket for storing the files (see below). When the experiment is commited and pushed, it is tracked in the bucket and can be utilized by all users (and after cleaning the cache). The data, models and experiments that are not commited and pushed can only be used locally, which means they are lost when the cached is cleaned.**

# Automate experiments using pipelines 
### Preparation 
- parameterize the pipeline by using a params.yaml file
- modularize pipeline steps in different files 
- adapt the code to utilize the params.yaml contents

### Create pipeline
- `dvc stage add -n stagename ...` adds a pipeline stage to the dvc.yaml file including the stage "train" with its parameters from the params.yaml file, dependencies and ouptuts as well as a command to run the pipeline

Example with the modified train.py:

```bash
dvc stage add -n train \
  --params base,train \
  --deps train.py --deps data/raw \
  --outs models/model.pth \
  python src/train.py
```
- `dvc dag` visualizes the pipeline in the terminal

### Run the pipeline
- `dvc exp run` runs the pipeline from the dvc.yaml file and captures the state of the workspace as DVC experiment 
- `dvc exp run --name "batch-size_8" --set-param "train.batch_size=8"` changes the batch_size to 8 and runs the experiment
- `dvc exp run --name "batch-size-tryout" --queue -S "train.batch_size=8,16,32"` queues 3 experiments with different batch sizes, to run: 
- `dvc exp run --run-all`

# Store Data in external Bucket 
TODO: 
## Generate  remote:
- ```dvc remote add -d remotename url:port/path``` adds config for remote datastore location ```dvc remote add -d heymates_remote ssh://login@serverip/path/to/datastore``` #Note: This needs to be the absolute path
- ```dvc remote modify remote_name ask_password true``` enables password auth at connection
- `dvc remote modify --local remote_name password yourpassword` adds config.local file with the password stored and adds this file to gitignore

## Upload Data:
- ```dvc add path/data.xml``` adds data to DVC (if not done earlier)
- ```git add path/data.xml.dvc data/.gitignore``` added den Bezug zu 
- ```git commit -m "Add raw data"``` tracked changes for dvc
- ```dvc push``` uploads the data to the remote store


## Notes:
- DVC stores data and experiments in the .dvc/cache/files/md5 directory 
- This means it copies (or links) the data and hashes it
- Only f BTRFS data management is available on your system, the data does not need to be copied (hardlink and symlink are not recommended due to the risk of data corruption)
- For experiment tracking, the dvc.yaml is copied to .dvc/cache/runs directory
- `dvc push` copies the cache to the remote 
- `dvc gc` cleans the local cache (deletes all files from cache other than the current working ones)
- `dvc checkout <branch-or-commit>` checks out a dedicated experiment