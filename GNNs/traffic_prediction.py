import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from traffic_dataset import LoadData  
from utils import Evaluation 
from utils import visualize_result
from gcnnet import GCN
from chebnet import ChebNet
from gat import GATNet

import warnings
warnings.filterwarnings('ignore')



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

    
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)  

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=32)

  
    my_net = GATNet(in_c=6 * 1, hid_c=6, out_c=1, n_heads=2)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    my_net = my_net.to(device)  

    
    criterion = nn.MSELoss()  

    optimizer = optim.Adam(params=my_net.parameters())


    Epoch = 2 

    my_net.train()  
    for epoch in range(Epoch):
        epoch_loss = 0.0
        count = 0
        start_time = time.time()
        for data in train_loader:  
            my_net.zero_grad()  
            count +=1
            predict_value = my_net(data, device).to(torch.device("cpu")) 

            loss = criterion(predict_value, data["flow_y"])  

            epoch_loss += loss.item()  

            loss.backward()  

            optimizer.step() 
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time - start_time) / 60))


    my_net.eval()  
    with torch.no_grad():  
        MAE, MAPE, RMSE = [], [], []
        Target = np.zeros([307, 1, 1])  
        Predict = np.zeros_like(Target)  

        total_loss = 0.0
        for data in test_loader: 

           
            predict_value = my_net(data, device).to(torch.device("cpu"))  

            loss = criterion(predict_value, data["flow_y"])  

            total_loss += loss.item() 
          
            predict_value = predict_value.transpose(0, 2).squeeze(0) 
            target_value = data["flow_y"].transpose(0, 2).squeeze(0)  

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)  

           
            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

            print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))


    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    Predict = np.delete(Predict, 0, axis=1) 
    Target = np.delete(Target, 0, axis=1)

    result_file = "GAT_result.h5"
    file_obj = h5py.File(result_file, "w")  

    file_obj["predict"] = Predict  
    file_obj["target"] = Target  


def compute_performance(prediction, target, data): 

    try:
        dataset = data.dataset  
    except:
        dataset = data  

 
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())


    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  


if __name__ == '__main__':
    main()

    visualize_result(h5_file="GAT_result.h5",
    nodes_id = 120, time_se = [0, 24 * 12 * 2],  
    visualize_file = "gat_node_120")
