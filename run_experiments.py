import json
from main import run_fl

file_obj = open('config.json','r')
config = json.load(file_obj) #Converts json file format into python dictionary
global_config = config["global_config"]
data_config = config["data_config"]
fed_config = config["fed_config"]
model_config = config["model_config"]

algorithms = ["FedAvg", "FedAvgM"] #["FedAvg", "FedAvgM", "FedAdam", "SCAFFOLD"]
non_iid = [0.55, 0.75] #[0, 0.2, 0.4, 0.55, 0.6, 0.75, 0.8, 0.9]

for algo in algorithms:
    fed_config["algorithm"] = algo
    exec(f"from src.algorithms.{fed_config['algorithm']}.server import Server")
    for n_i in non_iid:
        data_config["non_iid_per"] = n_i
        run_fl(Server, global_config, data_config, fed_config, model_config)

