import os
import json
import logging
import matplotlib.pyplot as plt
from src.util_functions import set_logger, save_plt

def run_fl(Server, global_config, data_config, fed_config, model_config):
    if not os.path.exists(f"./Logs/{fed_config['algorithm']}"):
        os.mkdir(f"./Logs/{fed_config['algorithm']}")
    if not os.path.exists(f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}"):
        os.mkdir(f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}")
    
    #Set Logger
    filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/"
    set_logger(f"{filename}log.txt")

    server = Server(model_config,global_config, data_config, fed_config)

    logging.info("Server is successfully initialized")
    server.setup() #Initializes all the clients and splits the train dataset among all the clients
    server.train() #Trains the global model for multiple rounds

    save_plt(list(range(1, server.num_rounds+1)),server.results['accuracy'],"Communication Round","Test Accuracy",f"{filename}accgraph.png")
    save_plt(list(range(1, server.num_rounds+1)),server.results['loss'],"Communication Round","Test Loss",f"{filename}lossgraph.png")
    logging.info("\nExecution has completed")

if __name__ == "__main__":
    file_obj = open('config.json','r')
    config = json.load(file_obj) #Converts json file format into python dictionary
    global_config = config["global_config"]
    data_config = config["data_config"]
    fed_config = config["fed_config"]
    model_config = config["model_config"]
    exec(f"from src.algorithms.{fed_config['algorithm']}.server import Server")
    run_fl(Server, global_config, data_config, fed_config, model_config)

