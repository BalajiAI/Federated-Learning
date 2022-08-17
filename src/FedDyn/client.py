import torch
from copy import deepcopy
from math import ceil

class Client():
    """
    Server uses Client class to create multiple client objects.
    Client trains the model  on its local data. Then the client updates its client_c 
    and communicates updates for the x (delta_y) back to the server.
    
    Attributes :
        id: Acts as an identifier for a particular client
        data: Local dataset which resides on the client
        device: Specifies which device (cpu or gpu) to use for training
        num_epochs:
        lr:
        criterion:
        x: Global model sent by server
        y: Local model initialized using x
        delta_y: Used to update the x    
    """
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr, model):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.alpha = 0.01
        self.criterion = criterion
        self.x = model
        self.y = None
        #delta_y of a client are communicated to the central server after client_update has completed
        self.delta_y = None
        
        self.prev_grads = None
        for param in self.x.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)
        

    def client_update(self):
        """
        Trains the model on its local data. Then calculates delta_y and delta_c which are communicated back to the server.
        At last, updates the client_c.
        """ 
        self.y = deepcopy(self.x) #Initialize local model 
        #self.y.to(device)
        
        for epoch in range(self.num_epochs):
            inputs,labels = iter(self.data).next()
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels
            
            #Dynamic Regularisation
            lin_penalty = 0.0
            curr_params = None
            for param in self.x.parameters():
                if not isinstance(curr_params, torch.Tensor):
                    curr_params = param.view(-1)
                else:
                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

            lin_penalty = torch.sum(curr_params * self.prev_grads)
            loss -= lin_penalty
            
            quad_penalty = 0.0
            for name, param in self.y.named_parameters():
                    quad_penalty += torch.nn.functional.mse_loss(param, self.x.state_dict()[name], reduction='sum')

            loss += (self.alpha/2) * quad_penalty

            gradients = torch.autograd.grad(loss,self.y.parameters())
            
            # Update the previous gradients
            self.prev_grads = None
            for grad in gradients:
                if not isinstance(self.prev_grads, torch.Tensor):
                    self.prev_grads = grad.view(-1).clone()
                else:
                    self.prev_grads = torch.cat((self.prev_grads, grad.view(-1).clone()), dim=0)

            for param, grad in zip(self.y.parameters(),gradients):
                param.data -= self.lr * grad.data
             

            #if self.device == "cuda": torch.cuda.empty_cache()