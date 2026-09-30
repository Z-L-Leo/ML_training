# ML_training

this is a repo for training models
# 1. create environment
Use this command to create environment
```sh
conda env create -f environment.yaml
conda activate trl
```

# 2.training a model
this command will train a model and store model in <code>./results</code>
```sh
python train.py
```

this command will train a model and push it to remote hugging face repo
```sh
python train_and_push.py
```

this command will profile the training process
```sh
python train_profiling.py
```
