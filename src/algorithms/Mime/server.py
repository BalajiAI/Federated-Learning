import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from copy import deepcopy

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn

class Server():
    """
    Server is initialized with set of hyperparameters, global model(x). Server then initializes a number of clients and 
    splits the train dataset among them. Server has only access the test dataset which is useful for evaluating the global 
    model's performance after each round. In each round, Server selects a fraction of clients and communicates x & server_c 
    to the selected clients. Each client then trains model(x) on its local dataset and communicates updates back to the 
    server. On receiving the updates from all the participating clients, server updates its x.
    
    Attributes:
        device: Specifies which device (cpu or gpu) to use for training
        data_path: Specifies the path for storing the dataset
        dataset_name: Specifies which datset to use
        fraction: Specifies what fraction of clients needs to be selected in each round
        num_clients: Total number of clients
        num_rounds: Number of rounds for training the global model
        num_epochs: Number of epochs to train the model on each client
        batch_size: Specifies the batch size for the test dataset
        criterion: Loss function to optimize 
        lr: Global stepsize which is used in server_update
        lr_l: Local stepsize which is used in client_update
        x: Global model which needs to be trained
        clients: List of clients whose length is equal to num_clients
    """
    
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
      
        set_seed(global_config["seed"])
        self.device = global_config["device"]

        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.fraction = fed_config["fraction_clients"]
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]
        self.beta = 0.9
        
        self.x = eval(model_config["name"])().to(self.device)   
        self.state = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]
        self.control_variate = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]
        
        self.clients = None       
    
    def create_clients(self, local_datasets):
        clients = []
        for id_num,dataset in enumerate(local_datasets):
            client = Client(client_id=id_num, local_data=dataset, device=self.device, num_epochs = self.num_epochs, criterion = self.criterion, lr=self.lr_l)
            clients.append(client)
        return clients
    
    def setup(self, **init_kwargs):
        """Initializes all the Clients and splits the train dataset among them""" 
        local_datasets,test_dataset = dist_data_per_client(self.data_path, self.dataset_name, self.num_clients, self.batch_size, self.non_iid_per, self.device)
        self.data = test_dataset
      
        self.clients = self.create_clients(local_datasets)
        logging.info("\nClients are successfully initialized")
      
    def sample_clients(self):
        """Selects a fraction of clients from all the available clients"""
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_ids = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, 
        replace=False).tolist())
        return sampled_client_ids

    def communicate(self, client_ids):
        """Communicates global model(x) to the participating clients"""
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)
            self.clients[idx].state = deepcopy(self.state)
            self.clients[idx].control_variate = deepcopy(self.control_variate)
               
    def update_clients(self, client_ids):
        """Tells all the clients to perform client_update"""
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """Updates the global model(x)"""
        avg_grads = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]
        delta_x = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]

        with torch.no_grad():
            for idx in client_ids:
                #Updates the x using the delta_y from all the clients
                for avg_grad, grad in zip(avg_grads, self.clients[idx].gradient_x):
                    avg_grad.data.add_(grad.data / int(self.fraction * self.num_clients))
                    
                for d_x, diff in zip(delta_x, self.clients[idx].delta_y):
                    d_x.data.add_(diff.data / int(self.fraction * self.num_clients))
            
            for s,grad in zip(self.state, avg_grads):
                s.data = (1-self.beta) * grad.data + self.beta * s.data
            
            self.control_variate = deepcopy(avg_grads)
            
            for param, d_x in zip(self.x.parameters(), delta_x):
                param.data = param.data + self.lr * d_x.data    
 
               
    def step(self):
        """Performs single round of training"""
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info("\tclient_update has completed") 
        self.server_update(sampled_client_ids)
        logging.info("\tserver_update has completed")

    def train(self):
        """Performs multiple rounds of training using the 'step' method."""
        self.results = {"loss": [], "accuracy": []}
        for round in range(self.num_rounds):
            logging.info(f"\nTraining Round:{round+1}")
            self.step()
            test_loss, test_acc = evaluate_fn(self.data,self.x,self.criterion,self.device) #Evaluates the global model's performance on the test datasat
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_acc)
            logging.info(f"\tLoss:{test_loss:.4f}   Accuracy:{test_acc:.2f}%")
            
