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
        self.control_variate = None
        self.y = None
        #delta_y of a client are communicated to the central server after client_update has completed
        self.gradient_x = None
        self.delta_y = None

    def client_update(self):
        """
        Trains the model on its local data. Then calculates delta_y and delta_c which are communicated back to the server.
        At last, updates the client_c.
        """ 
        self.y = deepcopy(self.x) #Initialize local model 
        self.y.to(self.device)
        
        for epoch in range(self.num_epochs):
            inputs, labels = iter(self.data).next()
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels
            #Compute (mini-batch) gradient of loss with respect to y's parameters 
            grads_y = torch.autograd.grad(loss,self.y.parameters())
            
            if (epoch == 0):
                output = self.x(inputs)
                loss = self.criterion(output, labels)
                grads_x = torch.autograd.grad(loss,self.x.parameters())
          
            with torch.no_grad():
                for g_y, g_x, c in zip(grads_y, grads_x, self.control_variate):
                    g_y.data -= g_x.data + c  
            
                for param,grad,s in zip(self.y.parameters(), grads_y, self.state):
                    param.data = param.data - self.lr * ((1-self.beta) * grad.data + self.beta * s.data)
        
            #if self.device == "cuda": torch.cuda.empty_cache()               
        #inputs,labels = iter(self.data).next()
        #inputs,labels = inputs.float().to(self.device), labels.long().to(self.device)
        #output = self.x(inputs)
        #loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels            
        #self.gradient_x = torch.autograd.grad(loss,self.x.parameters())
        self.gradient_x = grads_x
        
        with torch.no_grad():
            delta_y = [torch.zeros_like(param) for param in self.y.parameters()]
            #Calculate delta_y which equals to y-x [Algorithm line no:13]
            for del_y, param_y, param_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                del_y.data += param_y.data.detach() - param_x.data.detach()   
       
        self.delta_y = delta_y
        
