import torch
import epilearn as epi
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
        


if __name__ == "__main__":
    #with open("./papw/all_day_repeat20_stage1_5/total_infect_repeat20_stage1_5.pkl", "rb") as f:
    with open("./papw/total_infect/total_infect.pkl", "rb") as f:
        data = pickle.load(f)
    
    graph = np.load("./papw/graph.npy")
    #graph = np.load("./papw/location_graph_98.npy")
    #graph = np.vstack((graph, np.ones((1, graph.shape[1]))))
    #graph = np.hstack((graph, np.ones((graph.shape[0], 1))))
    
    data_x = data["x"]
    data_y = data["y"]
    print(f"data_x: {data_x.shape}, data_y: {data_y.shape}")
    horizon = 10
    pred_period = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    #surrogate_model = epi.models.SpatialTemporal.EpiGNN(num_nodes=11, num_features=1, num_timesteps_input=horizon, num_timesteps_output=pred_period,
    #                                                    device=device).to(device)
    
    
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
    
    train_data = data["all_data"][2:]
    mean = np.mean(train_data[:, :, 0:11])
    std = np.std(train_data[:, :, 0:11])
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
    
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(test_x.shape)


    surrogate_model.fit(train_input=torch.tensor(train_x[:, :, 0:11], dtype=torch.float).unsqueeze(-1),
                        train_target=torch.tensor(train_y[:, :, 0:11], dtype=torch.float),
                        train_graph=torch.tensor(graph, dtype=torch.float),
                        val_input=torch.tensor(val_x[:, :, 0:11], dtype=torch.float).unsqueeze(-1),
                        val_target=torch.tensor(val_y[:, :, 0:11], dtype=torch.float),
                        val_graph=torch.tensor(graph, dtype=torch.float),
                        epochs=3000,
                        lr=1e-4,
                        batch_size=128,
                        device=device
                        )

    if not os.path.exists("./papw/total_infect/pic/"):
        os.makedirs("./papw/total_infect/pic/")
        
    
    torch.save(surrogate_model.state_dict(), "./papw/total_infect/pic/best_surr_model.pth")
    
    
    
    surrogate_model_test = epi.models.SpatialTemporal.ColaGNN(num_nodes=11,
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
    
    surrogate_model_test.load_state_dict(torch.load("./papw/total_infect/pic/best_surr_model.pth", weights_only=True))
    surrogate_model_test.eval()
    all_data = data["all_data"]
    test_x = all_data

    #import ipdb; ipdb.set_trace()
    
    print("Infering on test set...")
    print(test_x.shape)
    for repeat_idx, test_data_x in tqdm(enumerate(test_x), total=len(test_x)):
        for stage_idx, stage in enumerate(range(test_data_x.shape[0])):
            if stage_idx == 2:
                continue
            data_x = test_data_x[stage_idx]
            for sld_win in range(0, data_x.shape[0], pred_period):
                
                if sld_win == 0:
                    y = torch.tensor((data_x[0:horizon, 0:11]-mean)/std, dtype=torch.float)
                    y = torch.cat((y, torch.sum(y, dim=-1).unsqueeze(-1)), dim=-1)
                    #print(y[-horizon:, :-1])
                #print(y.shape)
                #import ipdb; ipdb.set_trace()
                
                y_idx = surrogate_model_test.predict(y[-horizon:, 0:11].unsqueeze(0).unsqueeze(-1), graph=torch.tensor(graph, dtype=torch.float))
                #y_idx = surrogate_model.predict(torch.tensor(data_x[sld_win:sld_win+horizon, 0:11]/2000, dtype=torch.float).unsqueeze(0).unsqueeze(-1), graph=torch.tensor(graph, dtype=torch.float))
                y_idx = y_idx.squeeze()
                y_idx = torch.cat((y_idx, torch.sum(y_idx, dim=-1).unsqueeze(-1)), dim=-1)
                #print(y_idx.shape)
                if pred_period == 1:
                    y_idx = y_idx.unsqueeze(0)
                y = torch.cat((y, y_idx), dim=0)
                #y.append(y_idx.squeeze())
                #print(y.shape)
                
                if y.shape[0] >= all_data.shape[2]:
                    y = y[0:all_data.shape[2]]
                    break
            
            '''
            if stage_idx == 0:
                stage_y = y.unsqueeze(0)
            else:
                stage_y = torch.cat((stage_y, y.unsqueeze(0)), dim=0)
            '''
            
            true_y = test_x[repeat_idx, stage_idx]
            #print(true_y[:, -1])
            
            new = np.hstack((y[:, -1, np.newaxis]*std+mean*11, true_y[:, -1, np.newaxis]))
            layout = epi.visualize.plot_series(new, columns=['pred_Infect', 'Infect'])
            #plt.show()
            plt.savefig(f'./papw/total_infect/pic/simulated_repeat{repeat_idx+1}_stage{stage_idx+1}.png')
            plt.close()
            
            #exit()

    
    