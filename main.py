import datetime
import json
import logging
import matplotlib.pyplot as plt
from util_functions import set_logger, save_plt

with open('config.json','r') as file:
    config = json.load(file) #Converts json file format into python dictionary
global_config = config["global_config"]
data_config = config["data_config"]
fed_config = config["fed_config"]
model_config = config["model_config"]

#Set Logger
filename = f"./Logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
set_logger(f"{filename}_log.txt")

if (fed_config["algorithm"] == "fedavg"):
    from src.FedAvg.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
elif (fed_config["algorithm"] == "fedavgm"):
    from src.FedAvgM.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
elif (fed_config["algorithm"] == "fedadam"):
    from src.FedAdam.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
elif (fed_config["algorithm"] == "scaffold"):
    from src.SCAFFOLD.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
elif (fed_config["algorithm"] == "feddyn"):
    from src.FedDyn.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
elif (fed_config["algorithm"] == "mime"):
    from src.Mime.server import Server
    server = Server(model_config,global_config, data_config, fed_config)
else:
    raise AttributeError(f"{fed_config['algorithm']} algorithm is not found")

logging.info("Server is successfully initialized")
server.setup() #Initializes all the clients and splits the train dataset among all the clients
server.train() #Trains the global model for multiple rounds


save_plt(list(range(1, server.num_rounds+1)),server.results['accuracy'],"Num_rounds","Accuracy",f"{filename}_accgraph.png")
save_plt(list(range(1, server.num_rounds+1)),server.results['loss'],"Num_rounds","Loss",f"{filename}_lossgraph.png")

logging.info("\nExecution has completed")
