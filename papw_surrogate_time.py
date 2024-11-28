import torch
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
    
    data = np.delete(data, 2, axis=1)
    
    data_x = data[:, :, :, 0:11]
    data_y = np.expand_dims(np.sum(data, axis=3), axis=-1)
    print(f"data_x: {data_x.shape}, data_y: {data_y.shape}")
    horizon = 10
    pred_period = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    graph = torch.Tensor(graph).to(device)
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
    
    mean = np.mean(train_x)
    std = np.std(train_x)
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
    
    
    test_x = (test_x - mean) / std
    test_y = (test_y - mean) / std
    
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(test_x.shape)


    
    optimizer = torch.optim.AdamW(surrogate_model.parameters(), lr=1e-4)

    loss_fn = torch.nn.MSELoss()


    early_stopping = 10
    best_val = float('inf')
    epochs = 1000
    
    train_x = torch.Tensor(train_x).to(device)
    val_x = torch.Tensor(val_x).to(device)
    for epoch in range(epochs):
        print(f"-----Epoch {epoch+1}-----")
        epoch_loss = []
        for i in tqdm(range(train_x.shape[0])):
            data_x = train_x[i].clone()
            data_y = train_x[i]
            
            for sld_id in range(data_x.shape[0] - horizon - 1):
                data = data_x[sld_id:sld_id+horizon]
                label = data_x[sld_id+horizon+1]
                data = data.unsqueeze(0).unsqueeze(-1)
                #print(label)
                
                output = surrogate_model.forward(data, graph)
                data_x[sld_id+horizon+1] = output.squeeze()
                #print(output)

            loss = loss_fn(data_x, data_y)
            epoch_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch+1} Avg Training Loss: {epoch_avg_loss}")
        
        
        
        
        
        
        epoch_loss_val = []
        for i in tqdm(range(val_x.shape[0])):
            data_x = val_x[i].clone()
            data_y = val_x[i]
            
            for sld_id in range(data_x.shape[0] - horizon - 1):
                data = data_x[sld_id:sld_id+horizon]
                label = data_x[sld_id+horizon+1]
                data = data.unsqueeze(0).unsqueeze(-1)
                #print(label)
                
                output = surrogate_model.forward(data, graph)
                data_x[sld_id+horizon+1] = output.squeeze()
                #print(output)

            loss = loss_fn(data_x, data_y)
            epoch_loss_val.append(loss)
            loss.backward()
            optimizer.step()
        val_loss = sum(epoch_loss_val) / len(epoch_loss_val)
        print(f"Epoch {epoch+1} Avg Validation Loss: {val_loss}")


        if best_val > val_loss:
            best_val = val_loss
            patience = early_stopping
            best_model = surrogate_model
            print("The Best Epoch!")
        else:
            patience -= 1

        if epoch > early_stopping and patience <= 0:
            break
    
    
    
    

    if not os.path.exists("./papw_non/total_infect/pic/"):
        os.makedirs("./papw_non/total_infect/pic/")
        
    
    torch.save(surrogate_model.state_dict(), "./papw_non/total_infect/pic/best_surr_model.pth")
    
    
    
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
    
    surrogate_model_test.load_state_dict(torch.load("./papw_non/total_infect/pic/best_surr_model.pth", weights_only=True))
    surrogate_model_test.eval()
    

    #import ipdb; ipdb.set_trace()
    
    print("Infering on test set...")
    print(test_x.shape)
    for repeat_idx, test_data_x in tqdm(enumerate(test_x), total=len(test_x)):
        for stage_idx, stage in enumerate(range(test_data_x.shape[0])):
            
            data_x = test_data_x[stage_idx]
            print(data_x.shape)
            for sld_win in range(0, data_x.shape[0], pred_period):
                
                if sld_win == 0:
                    print(data_x[0:horizon, 0:11])
                    y = torch.tensor(data_x[0:horizon, 0:11], dtype=torch.float).clone()
                    y = torch.cat((y, torch.sum(y, dim=-1).unsqueeze(-1)), dim=-1).clone()
                    #print(y[-horizon:, :-1])
                #print(y.shape)
                #import ipdb; ipdb.set_trace()
                
                y_idx = surrogate_model_test.predict(y[-horizon:, 0:11].unsqueeze(0).unsqueeze(-1).clone(), graph=graph).clone()
                #y_idx = surrogate_model.predict(torch.tensor(data_x[sld_win:sld_win+horizon, 0:11]/2000, dtype=torch.float).unsqueeze(0).unsqueeze(-1), graph=torch.tensor(graph, dtype=torch.float))
                y_idx = y_idx.squeeze()
                y_idx = torch.cat((y_idx, torch.sum(y_idx, dim=-1).unsqueeze(-1)), dim=-1).clone()
                #print(y_idx.shape)
                if pred_period == 1:
                    y_idx = y_idx.unsqueeze(0)
                y = torch.cat((y, y_idx), dim=0)
                #y.append(y_idx.squeeze())
                #print(y.shape)
                
                if y.shape[0] >= test_x.shape[2]:
                    y = y[0:test_x.shape[2]].clone()
                    break
            
            '''
            if stage_idx == 0:
                stage_y = y.unsqueeze(0)
            else:
                stage_y = torch.cat((stage_y, y.unsqueeze(0)), dim=0)
            '''
            
            true_y = torch.tensor(test_x[repeat_idx, stage_idx])
            true_y = torch.cat((true_y, torch.sum(true_y, dim=-1).unsqueeze(-1)), dim=-1)
            print(true_y)
            
            new = np.hstack((y[:, -1, np.newaxis]*std+mean*11, true_y[:, -1, np.newaxis]*std+mean*11))
            layout = epi.visualize.plot_series(new, columns=['pred_Infect', 'Infect'])
            #plt.show()
            if stage_idx > 1:
                plt.savefig(f'./papw_non/total_infect/pic/simulated_repeat{repeat_idx+1}_stage{stage_idx+2}.png')
            else:
                plt.savefig(f'./papw_non/total_infect/pic/simulated_repeat{repeat_idx+1}_stage{stage_idx+1}.png')
            plt.close()
            
            #exit()

    
    