import torch
import torch.nn as nn
import torch.nn.functional as F
import epilearn as epi
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 

        
if __name__ == "__main__":
    data = np.load("./papw_non/total_infect/total_infect.npy")
    print(data.shape)
    
    
    graph = np.load("./papw/graph.npy")
    print(graph.shape)
    
    
    data = np.delete(data, 2, axis=1)
    
    data_x = data[:, :, :, 0:11]
    data_y = np.expand_dims(np.sum(data, axis=3), axis=-1)
    
    print(f"data_x: {data_x.shape}, data_y: {data_y.shape}")
    
    horizon = 10
    pred_period = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)


    from sklearn.model_selection import train_test_split
    test_x = data_x[0:2]
    test_y = data_y[0:2]

    
    train_x, val_x, train_y, val_y = train_test_split(data_x[2:], data_y[2:], test_size=0.2)
    #train_x, val_x, train_y, val_y = data_x[0:1], data_x[0:1], data_y[0:1], data_y[0:1]
    print(train_x.shape)
    print(val_x.shape)
    print(test_x.shape)
    
    ##############
    #train_x = train_x[:, 0, :, :, :]
    #val_x = val_x[:, 0, :, :, :]
    ##############
    
    train_x = train_x.reshape(-1, train_x.shape[-2], train_x.shape[-1])
    train_y = train_y.reshape(-1, train_y.shape[-2], train_y.shape[-1])
    
    
    mean = np.mean(np.concatenate((data_x, data_y), axis=3))
    std = np.std(np.concatenate((data_x, data_y), axis=3))
    print(f"mean: {mean} - std: {std}")
    
    train_x = (train_x - mean) / std
    train_y = (train_y - mean) / std
    
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]
    
    val_x = val_x.reshape(-1, val_x.shape[-2], val_x.shape[-1])
    val_y = val_y.reshape(-1, val_y.shape[-2], val_y.shape[-1])
    
    val_x = (val_x - mean) / std
    val_y = (val_y - mean) / std
    
    indices = np.arange(val_x.shape[0])
    np.random.shuffle(indices)
    val_x = val_x[indices]
    val_y = val_y[indices]
    
    #test_x = test_x.reshape(-1, train_x.shape[-2], train_x.shape[-1])
    test_x = (test_x - mean) / std
    test_x = (test_x - mean) / std
    
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(test_x.shape)
    
    surrogate_model = epi.models.SpatialTemporal.ColaGNN(num_nodes=11,
                num_features=1,
                num_timesteps_input=horizon,
                num_timesteps_output=pred_period,
                nhid=256,
                n_channels=8,
                rnn_model = 'LSTM',
                n_layer = 4,
                bidirect = False,
                dropout = 0.5,
                device=device).to(device)
    
    surrogate_model.load_state_dict(torch.load("./papw/total_infect/pic/best_surr_model.pth", weights_only=True))
    surrogate_model.train()
    
    num_days = 60
    strategy_vector = torch.rand(60, 11, requires_grad=True, device=device)

    optimizer = torch.optim.AdamW([strategy_vector], lr=1e-4)
    graph_seq = np.repeat(graph[np.newaxis, :, :], 10, axis=0)
    
    graph = torch.tensor(graph, dtype=torch.float).to(device)
    graph_seq = torch.tensor(graph_seq, dtype=torch.float).to(device)

    best_loss = torch.inf
    print("Start training...")
    print("-"*20)
    print(train_x.shape)
    tolerance = 0
    
    if not os.path.exists("./papw/total_infect/modified/"):
        os.makedirs("./papw/total_infect/modified/")
    
    epochs = 1000
    for epoch in range(0, epochs):
        
        day_pbar = tqdm(range(10, num_days), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        modified_graph = graph
        loss_list = []
        for day in day_pbar:
            current_data = train_x[0, day, :]
            if day < 10:
                continue

            beta = 10
            strategies_up_to_current_day = strategy_vector[:day + 1]  # [day + 1, num_nodes]

            # 计算每个节点的因子
            factors = torch.sigmoid(beta * (strategies_up_to_current_day - 0.5))  # [day + 1, num_nodes]

            # 计算每个节点的累积因子
            log_factors = torch.log(1 - factors + 1e-10)  # [day + 1, num_nodes]
            log_cumulative_factors = torch.sum(log_factors, dim=0)  # [num_nodes]
            cumulative_factors = torch.exp(log_cumulative_factors)  # [num_nodes]
            
            cumulative_factors = cumulative_factors.unsqueeze(dim=-1)
            
            modified_graph = modified_graph * (cumulative_factors.T * cumulative_factors)

            next_day = surrogate_model(torch.Tensor(train_x[0, day-10:day, :]).unsqueeze(0).unsqueeze(-1).to(device), modified_graph)
            
            #print(next_day.shape)
            
            if day < num_days - 1:
                train_x[0, day+1, :] = next_day.detach().cpu().numpy()

            current_loss = next_day.sum()

            loss_list.append(current_loss)

            # 更新进度条
            day_pbar.set_postfix({'Day Loss': int(current_loss.item())})

        day_pbar.close()
        
        accumulated_isolation = torch.zeros_like(strategy_vector[0])  # 初始累积隔离概率

        for day, strategy in enumerate(strategy_vector):
            if day < 10:
                continue
            # 使用 sigmoid 函数近似每日隔离决策
            current_isolation = torch.sigmoid(beta * (strategy - 0.5))
            # 更新累积隔离概率
            accumulated_isolation = 1 - (1 - accumulated_isolation) * (1 - current_isolation)

        # 累积隔离概率的总和即为平滑的隔离节点数
        iso_num = torch.sum(accumulated_isolation)
        total_infec = torch.stack(loss_list).sum()
        epoch_loss = total_infec + 80 * iso_num / num_days
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {int(epoch_loss.item())}, "
                f"Infections per Day: {int(total_infec.item() / num_days)}, "
                f"Total Isolated Nodes: {int(iso_num.item())}")
        
