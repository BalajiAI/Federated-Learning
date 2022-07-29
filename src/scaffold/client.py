import torch
from copy import deepcopy
from math import ceil

class Client():
    """
    Server uses Client class to create multiple client objects.
    Client trains the model with the help of server_c & client_c on its local data. Then the client updates its client_c 
    and communicates updates for both the x (delta_y) & server_c (delta_c) back to the server.
    
    Attributes :
        id: Acts as an identifier for a particular client
        data: Local dataset which resides on the client
        device: Specifies which device (cpu or gpu) to use for training
        num_epochs:
        lr:
        criterion:
        x: Global model sent by server
        server_c: Server's Control variate
        client_c: Client's Control variate
        y: Local model initialized using x
        delta_y: Used to update the x
        delta_c: Used to update the server_c     
    """
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr, client_c):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None
        self.server_c = None
        #Each client has its own control variate named client_c
        self.client_c = client_c
        self.y = None
        #delta_y & delta_c of a client are communicated to the central server after client_update has completed
        self.delta_y = None
        self.delta_c = None

    def client_update(self):
        """
        Trains the model on its local data. Then calculates delta_y and delta_c which are communicated back to the server.
        At last, updates the client_c.
        """ 
        self.y = deepcopy(self.x) #Initialize local model [Algorithm line no:7]
        #self.y.to(device)
        
        for epoch in range(self.num_epochs):
            inputs,labels = iter(self.data).next()
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels) #Calculate the loss with respect to y's output and labels
            #Compute (mini-batch) gradient of loss with respect to y's parameters [Algorithm line no:9]
            grads = torch.autograd.grad(loss,self.y.parameters())
                
            #Update y's parameters using gradients, client_c and server_c [Algorithm line no:10]
            with torch.no_grad():
                for param,grad,s_c,c_c in zip(self.y.parameters(),grads,self.server_c,self.client_c):
                    s_c, c_c = s_c.to(self.device), c_c.to(self.device)
                    #print(param)
                    param.data = param.data - self.lr * (grad.data + (s_c.data - c_c.data))
                    #print(param)
           
            #if self.device == "cuda": torch.cuda.empty_cache()               
       
        with torch.no_grad():

            delta_y = [torch.zeros_like(param) for param in self.y.parameters()]
            delta_c = deepcopy(delta_y)
            new_client_c = deepcopy(delta_y)

            #Calculate delta_y which equals to y-x [Algorithm line no:13]
            for del_y, param_y, param_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                del_y.data += param_y.data.detach() - param_x.data.detach()    
                # doubt : does x-y and y-x gives the same results?

            #Calculate new_client_c using client_c, server_c and delta_y [Algorithm line no:12]
            a = (ceil(len(self.data.dataset) / self.data.batch_size)*self.num_epochs*self.lr)
            for n_c, c_l, c_g, diff in zip(new_client_c, self.client_c, self.server_c, delta_y):
                n_c.data += c_l.data - c_g.data - diff.data / a
                    
            #Calculate delta_c which equals to new_client_c-client_c [Algorithm line no:13]
            for d_c, n_c_l, c_l in zip(delta_c, new_client_c, self.client_c):
                d_c.data.add_(n_c_l.data - c_l.data)
                
                
        self.client_c = deepcopy(new_client_c) #Update client_c with new_client_c
        self.delta_y = delta_y
        self.delta_c = delta_c
        
