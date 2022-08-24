import os
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:

    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.handlers.clear()

    logger.setLevel(logging.INFO)    
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
      
def save_plt(x,y,xlabel,ylabel,filename):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data(data_path, dataset_name):
    """This function downloads the specified the dataset"""
    
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if (dataset_name == "CIFAR10"):
            T = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
        elif (dataset_name == "MNIST"):
            T = transforms.ToTensor()
        
        
        train_data = datasets.__dict__[dataset_name](root=data_path, train=True, download=True,    
                                                                        transform=T)
        test_data = datasets.__dict__[dataset_name](root=data_path, train=False, download=True,  
                                                                    transform=T)
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)
    
    # unsqueeze channel dimension for grayscale image datasets
    if train_data.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        train_data.data.unsqueeze_(3)
 
    return train_data,test_data

# This class is used to get data in batches
class load_data(Dataset):
    def __init__(self, x, y):
        self.length = x.shape[0]
        self.x = x.permute(0,3,1,2)
        self.y = y
        self.image_transform = transforms.Normalize((127.5, 127.5, 127.5),(127.5, 127.5, 127.5))
        
    def __getitem__(self, index):
        image,label = self.x[index],self.y[index]
        image = self.image_transform(image)
        return image,label
        
    def __len__(self):
        return self.length
           
# A simple utility function for converting pytorch tensors to numpy
def tensor_to_numpy(data, device):
    if device.type == "cpu":
        return data.detach().numpy()
    else:
        return data.cpu().detach().numpy()

# A simple utility function for converting numpy to pytorch tensors
def numpy_to_tensor(data, device, dtype="float"):
    if dtype=="float":
        return torch.tensor(data, dtype=torch.float).to(device)
    elif dtype=="long":
        return torch.tensor(data, dtype=torch.long).to(device)

#Evaluates the model on a given dataset
def evaluate_fn(dataloader,model,loss_fn,device):

    model.eval()

    running_loss = 0
    total = 0
    correct = 0

    for batch,(images,labels) in enumerate(dataloader):
        output = model(images.to(device))
        loss = loss_fn(output,labels.to(device))
      
        running_loss += loss.item()

        total += labels.size(0)
        correct += (output.argmax(dim=1).cpu().detach() == labels.cpu().detach()).sum().item()
   
    avg_loss = running_loss/(batch+1)
    acc = 100*(correct/total)
    return avg_loss,acc
