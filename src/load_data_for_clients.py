import math
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.util_functions import set_seed, create_data, numpy_to_tensor, load_data


# Return num_clients+1 data. The +1 is for the server test set to evaluate performance
# non_iid_per - Define the amount of non-iid-ness required (range is between 0 and 1)
def dist_data_per_client(data_path,dataset_name,num_clients, batch_size, non_iid_per, device):
    set_seed(27)
    print("\nPreparing Data")
    train_data,test_data = create_data(data_path,dataset_name)
    X_train = train_data.data
    Y_train = np.array(train_data.targets)
    
    # Convert to numpy
    X_train = np.array(X_train)

    print("\nDividing the data among clients")

    # Find all the class labels
    classes = list(np.unique(Y_train))
    classes.sort()

    # Private variable.
    # Defines the step-size above which the inter-chunk non-iid increases
    step = math.ceil(100/len(classes))
    # Private variable
    # Defines the number of clients in each chunk
    min_client_in_chunk = 10 #70

    # This will contain the feature matrix for each clients
    client_data_feats = [list() for i in range(num_clients)]
    # This will contain the labels for each clients
    client_data_labels = [list() for i in range(num_clients)]

    # This defines the amount of non-iid in terms of step size
    inter_non_iid_score = int((non_iid_per*100)/step)
    intra_non_iid_score = int((non_iid_per*100)%step)

    # This loop is used to find which chunk receives which classes
    class_chunks = list()
    tmp = list()
    for index, class_ in enumerate(classes):
        # Since class labels will always be sequential, hence generate a list of class labels for current chunk
        # The mod is used to make sure that if we cross the number of classes then we go to the beginning of class array
        indices = np.arange(index,index+inter_non_iid_score)%len(classes)
        # Remove these generated classes from the actual class list
        class_chunk = list(set(classes) - set(np.array(classes)[indices]))
        # Since set is unsorted, sort the results back
        class_chunk.sort()
        # Now this chunk has the classes it requires
        class_chunks.append(class_chunk)
        tmp.extend(class_chunk)
    val = Counter(tmp)[classes[0]]

    total_clients = num_clients
    clients_per_chunk = list()
    # Randomly assign number of clients to each chunk (all chunks get different number of clients)
    # Current chunk gets number of clients samples from a uniform random distribution with
    # low = minimum_clients_per_chunk
    # high = current_total_clients - minimum_clients_per_chunk * total_number_of_chunks - (current_chunk_index + 1)
    # Then reduce current_total_clients by the number of clients selected for current chunk
    for i in range(len(class_chunks)):        
        clients_per_chunk.append(random.randint(min_client_in_chunk, total_clients - min_client_in_chunk*(len(class_chunks)-i-1)))
        total_clients-= clients_per_chunk[-1]
    print(clients_per_chunk)
    # This variable is required for indexing purposes
    cumulative_clients_per_chunk = [sum(clients_per_chunk[:i+1]) for i in range(len(clients_per_chunk))]

    # Give each class a 0 index. If a class is shared among chunks then its samples should be divided among the chunks
    # This counter helps in dividing the classes
    class_count_dict = dict([[class_, 0] for class_ in classes])
    # Now we assign samples to each chunk for the corresponding classes (Inter-chunk non-iid)
    #for index, class_chunk in tqdm(enumerate(class_chunks), total=len(classes)):
    for index, class_chunk in enumerate(class_chunks):
        # Run a loop for each class
        for class_label in class_chunk:
            # Find all the indices where the current class exists
            indices = np.where(Y_train == class_label)[0]
            # If the class is shared among client chunks, then divide it up
            start = round(class_count_dict[class_label]*(len(indices)/Counter(tmp)[class_label]))
            end = round((class_count_dict[class_label]+1)*(len(indices)/Counter(tmp)[class_label]))
            # Increase the counter so that the next chunk receives a different partition of samples for the same class
            class_count_dict[class_label]+=1
            # Take only those indices which belong to the current partition
            indices = indices[start:end]
            # Find number of samples per client
            num_data_per_client = math.ceil(len(indices)/clients_per_chunk[index])
            last_client_data = len(indices)%clients_per_chunk[index]

            # Generate a straight line
            # We will use this straight line to provide each client having same class different number of samples
            # val_last_client is the minimum sample the last client in a chunk gets
            val_last_client = 5
            # x1 and x2 are simply the 1st and last client in a chunk
            x1, x2 = 1, clients_per_chunk[index]
            # y1 is the maximum value the first client gets while y2 is the minimum value the last client gets
            y1, y2 = num_data_per_client+last_client_data-val_last_client, val_last_client
            # The first case is when slope is 0
            min_m, min_c = 0, val_last_client
            # Maximum non-iid is when slope is negative
            max_m = (y2-y1)/(x2-x1)
            max_c = y1-(max_m*x1)
            # Get the slope and intercept based on the amount of non-iid
            # We interpolate between the max and min slopes and intercepts in order to get the current amount of non-iid
            m = min_m + (((max_m - min_m)/(x2-x1))*intra_non_iid_score)
            c = min_c + (((max_c - min_c)/(x2-x1))*intra_non_iid_score)
            agg_points = 0

            # The values are then normalized and multiplied with the number of samples in a chunk
            # To make sure that the sample distribution within a chunk is equal to the number of samples that the entire chunk is allocated
            denom = sum([m*(i+1) + c for i in range(clients_per_chunk[index])])
            weights = [(m*(i+1) + c)/denom for i in range(clients_per_chunk[index])]
            
            client_index_start = cumulative_clients_per_chunk[index-1] if index > 0 else 0
            client_index_end = cumulative_clients_per_chunk[index]
            # Now assign the samples in current chunk to the clients that belong to this chunk
            for index_count, i in enumerate(np.arange(client_index_start, client_index_end)):
                if i >=num_clients:
                    break
                else:
                    # Each client gets a different number of samples as per the non-iid value
                    num_points = weights[index_count]*len(indices)
                    # Each client gets only a predfined number of samples
                    data = X_train[indices[round(agg_points):round(agg_points+num_points)]]
                    # Each client gets only a predfined number of labels of the current class
                    labels = [class_label for j in range(len(data))]
                    # Add the data to the client list
                    client_data_feats[i].extend(data)
                    # Add the labels to the client list
                    client_data_labels[i].extend(labels)
                    agg_points+= num_points

    # Now create the data loader object
    client_loaders = list()
    for i in range(num_clients):
        x = numpy_to_tensor(np.asarray(client_data_feats[i]), device, "float")
        y = numpy_to_tensor(np.asarray(client_data_labels[i]), device, "long")
        dataset = load_data(x, y)
        client_loaders.append(DataLoader(dataset=dataset, batch_size=x.shape[0], shuffle=True, num_workers=0))

    # Finally create the server test set
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return client_loaders, test_loader

#Testing purposes

#client_loaders, test_loader = dist_data_per_client("Data/CIFAR10","CIFAR10",200, 32, 0.65, "cpu")

'''

print(len(client_loaders))
print(client_loaders[0])
for i in client_loaders:
    print(len(i))
    x,y = next(iter(i))
    print(x.shape[0])
 
'''

