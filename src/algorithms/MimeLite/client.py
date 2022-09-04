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
        num_epochs: Number of epochs to train the local model
        lr: Local stepsize
        criterion: Measures the disagreement between model's prediction and ground truth
        x: Global model sent by server
        y: Local model initialized using x
        delta_y: Used to update the x    
    """
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta = 0.9
        self.criterion = criterion
        self.x = None
        self.state = None
        self.y = None
        
        self.gradient_x = None      

    def client_update(self):
        """
        Trains the model on its local data. Then calculates delta_y and delta_c which are communicated back to the server.
        At last, updates the client_c.
        """ 
        self.y = deepcopy(self.x) #Initialize local model 
        self.y.to(self.device)
        
        for epoch in range(self.num_epochs):
            inputs,labels = iter(self.data).next()
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels
            
            #Compute (full-batch) gradient of loss with respect to y's parameters 
            grads = torch.autograd.grad(loss,self.y.parameters())
            #Update y's parameters using gradients
            with torch.no_grad():
                for param,grad,s in zip(self.y.parameters(), grads, self.state):
                    param.data = param.data - self.lr * ((1-self.beta) * grad.data + self.beta * s.data)

            #if self.device == "cuda": torch.cuda.empty_cache()               
       
        inputs,labels = iter(self.data).next()
        inputs,labels = inputs.float().to(self.device), labels.long().to(self.device)
        output = self.x(inputs)
        loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels            
        self.gradient_x = torch.autograd.grad(loss,self.x.parameters())