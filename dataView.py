import numpy as np
import matplotlib.pyplot as plt


def get_flow(file_name):  

    flow_data = np.load(file_name) 
    print([key for key in flow_data.keys()])  

    print('before flow_data',flow_data["data"].shape)  
    flow_data = flow_data['data'].transpose([1, 0, 2])[:,:,0][:,:,np.newaxis]
    return flow_data

def pre_process_data(data, norm_dim): 

    norm_base = normalize_base(data, norm_dim) 
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  

    return norm_base, norm_data  


def normalize_base(data, norm_dim):

    max_data = np.max(data, norm_dim, keepdims=True)  
    min_data = np.min(data, norm_dim, keepdims=True)

    return max_data, min_data   


def normalize_data(max_data, min_data, data):

    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / base

    return normalized_data

def slice_data(data, history_length, index, train_mode): 

    if train_mode == "train":
        start_index = index 
        end_index = index + history_length 
    elif train_mode == "test":
        start_index = index - history_length 
        end_index = index 
    else:
        raise ValueError("train model {} is not defined".format(train_mode))

    print('data',data.shape)
    data_x = data[:, start_index: end_index]  
    data_y = data[:, end_index]  

    return data_x, data_y


if __name__ == "__main__":
    traffic_data = get_flow("PeMS_04/PeMS04.npz")

    norm_base, norm_data = pre_process_data(traffic_data,1)
    data_x, data_y = slice_data(norm_data,5,100,'train')

  
