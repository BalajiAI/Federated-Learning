# Federated-Learning 
This repository contains the code for Federated Learning algorithms such as [**FedAvg**](https://arxiv.org/abs/1602.05629), [**FedAvgM**](https://arxiv.org/abs/1909.06335), [**FedAdagrad**](https://arxiv.org/abs/2003.00295), [**FedAdam**](https://arxiv.org/abs/2003.00295), [**FedYogi**](https://arxiv.org/abs/2003.00295), [**SCAFFOLD**](https://arxiv.org/abs/1910.06378), [**MimeLite**](https://arxiv.org/abs/2008.03606) which are all implemented using PyTorch.

## Execution
You can run this code by executing `python main.py` command in your terminal. You can modify the **_config.json_** file to match the set of hyperarameters that you wish to use for execution.

## Running Experiments
If you wish to run multiple experiments on the algorithms by changing various hyperparameters, then use `python run_experiments.py` command. You can also modify the **_run_experiments.py_** file, to specify which algorithm to run and what set of hyperparameters to use for the experiments.
