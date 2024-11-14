import torch
import torch.nn as nn
import torch.nn.functional as F
import epilearn as epi
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 


class attack_model(nn.Module):
    def __init__(self, node_num=11, device="cpu", horizon=10):
        super(attack_model, self).__init__()
        self.model_name = "vector"
        self.device = device
        self.horizon = horizon
        
        #self.encoder = nn.Linear(165, node_num)
        self.encoder = nn.LSTM(self.horizon, node_num)#[5,11]
        self.learnable_vector_1 = nn.Linear(node_num, node_num)
        self.learnable_vector_2 = nn.Linear(node_num, 1)
        
        self.learnable_matrix = nn.Parameter(torch.randn(node_num, node_num))
        self.mask_layer = nn.Sigmoid()
        
        
        
        
    def forward(self, x, graph, graph_seq):
        #x = x.view(1, 165)
        #x = self.encoder(x)
        #print(x.shape)
        x,_ = self.encoder(torch.permute(x.squeeze()[-self.horizon:, :], (1, 0)))
        #x,_ = self.encoder(torch.permute(x.squeeze(), (0, 2, 1)))
        #print(x.shape)
        #modified_graph = self.mask_layer(torch.matmul(self.learnable_vector, graph))
        graph_mask = self.learnable_vector_1(x)
        graph_mask = self.learnable_vector_2(graph_mask)
        #import ipdb;ipdb.set_trace()
        
        #print("mask: ",self.mask_layer(graph_mask))
        graph_mask = self.mask_layer(graph_mask)
        #print(graph_mask.shape)
        
        graph_diag = torch.zeros_like(graph, dtype=bool).fill_diagonal_(True)
        modified_graph = graph
        #print((graph_mask.T * graph_mask).shape)
        modified_graph = modified_graph * (graph_mask.T * graph_mask)
        modified_graph[graph_diag] = graph[graph_diag]
        #print(modified_graph)
        
        #print(graph)
        #print(modified_graph)
        
        
        
        #mean_graph = torch.mean(graph_seq, axis=0)
        #modified_graph = mean_graph * self.learnable_matrix
        
        
        
        return modified_graph, graph_mask

'''
class attack_vec_model(nn.Module):
    def __init__(self, node_num=11, device="cpu"):
        super(attack_vec_model, self).__init__()
        self.vector = torch.nn.Parameter(torch.randn(node_num))
        self.softmax = F.softmax()
    
    def forward(self, x, graph):
        graph_diag = torch.zeros_like(graph, dtype=bool).fill_diagonal_(True)
        modified_graph = graph
        #print(tself.vector.unsqueeze(0) * self.vector.unsqueeze(1))
        modified_graph = modified_graph * self.softmax(self.vector).unsqueeze(0) * self.softmax(self.vector).unsqueeze(1)
        modified_graph[graph_diag] = graph[graph_diag]
        return modified_graph
'''   
    
def train_epoch(x, model, optimizer, surrogate_model, graph, graph_seq, device):
    model.train()
    optimizer.zero_grad()
    modified_graph, graph_mask = model(x, graph, graph_seq)
    
    
    #import ipdb; ipdb.set_trace()
    #print("y: ", y)
    #print("pred: ", pred_y)
    #print(graph_mask)
    
    surrogate_model.train()
    #surrogate_model.eval()
    # import ipdb; ipdb.set_trace()
    y = surrogate_model.forward(x, graph)
    pred_y = surrogate_model.forward(x, modified_graph)
    #print(y, pred_y)
    
    #loss = -(pred_y.sum() - y.sum())**2 + F.relu(pred_y.sum() - y.sum()) #+ F.relu(-pred_y.sum()))
    loss = pred_y.sum() #+ (1 - graph_mask).sum() #+ F.relu(pred_y.sum() - y.sum())
    #loss = pred_y.sum() + F.relu(pred_y.sum() - y.sum())**2 + F.relu(-pred_y.sum()) #-(pred_y.sum() - y.sum())**2 + pred_y.sum() * F.relu(pred_y.sum() - y.sum()) #+ F.relu(-pred_y.sum()))
    #print("loss: ", loss)
    #loss = pred_y.sum()
    loss.backward()
    #print("grad: ", model.learnable_vector.weight.grad)
    
    surrogate_model.eval()
    optimizer.step()
    
    return model, loss


def val_epoch(x, model, surrogate_model, graph, graph_seq):
    model.eval()
    modified_graph, graph_mask = model(x, graph, graph_seq)
    y = surrogate_model.forward(x, graph)
    pred_y = surrogate_model.forward(x, modified_graph)
    
    loss = pred_y.sum() #+ (1 - graph_mask).sum() # F.relu(pred_y.sum() - y.sum())
        
    return loss
        
        
        
if __name__ == "__main__":
    #with open("./papw/all_day_repeat20_stage1_5/all_day_repeat20_stage1_5_padding.pkl", "rb") as f:
    with open("./papw/total_infect/total_infect.pkl", "rb") as f:
        data = pickle.load(f)
    
    graph = np.load("./papw/graph.npy")
    print(graph.shape)
    #graph = np.vstack((graph, np.ones((1, graph.shape[1]))))
    #graph = np.hstack((graph, np.ones((graph.shape[0], 1))))
    print(graph.shape)
    
    data_x = data["x"]
    data_y = data["y"]
    print(f"data_x: {data_x.shape}, data_y: {data_y.shape}")
    horizon = 10
    pred_period = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    #surrogate_model = epi.models.SpatialTemporal.EpiGNN(num_nodes=11, num_features=1, num_timesteps_input=horizon, num_timesteps_output=pred_period, device=device).to(device)

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
    surrogate_model.eval()
    
    
    model = attack_model(node_num=11, device=device, horizon=horizon//2).to(device)
    #model = attack_vec_model(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
    
    for epoch in range(0, 5):
        
        loss_epoch = []
        
        for idx, x in tqdm(enumerate(train_x), total=len(train_x)):
        #for idx in tqdm(range(0, len(train_x), batch_size), total=len(train_x)//batch_size):
            #x = train_x[idx:idx+batch_size]
            #print(x.shape)
            model, loss = train_epoch(torch.tensor(x[:, :11], dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device), model, optimizer, surrogate_model, graph, graph_seq, device)
            #model, loss = train_epoch(torch.tensor(x[:, :, :11], dtype=torch.float).unsqueeze(-1).to(device), model, optimizer, surrogate_model, graph, device)
        
            loss_epoch.append(loss)
        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        print(f"Epoch {epoch+1} - train loss: {loss_epoch}")
        
        #exit()
        
        val_loss_epoch =[]
        for idx, x in tqdm(enumerate(val_x), total=len(val_x)):
            val_loss = val_epoch(torch.tensor(x[:, :11], dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device), model, surrogate_model, graph, graph_seq)
            val_loss_epoch.append(val_loss)
                
        
        val_loss_epoch = sum(val_loss_epoch) / len(val_loss_epoch)
        print(f"Epoch {epoch+1} - val loss: {val_loss_epoch}")
        if best_loss > val_loss_epoch:
            print(f"Best Epoch {epoch+1} - val loss: {val_loss_epoch}")
            best_epoch = epoch
            best_loss = val_loss_epoch
            best_model = model
            tolerance = 0
            torch.save(best_model.state_dict(), "./papw/total_infect/modified/best_attack_model.pth")
        else:
            tolerance += 1
        if tolerance >= 5:
            print("Early Stopped!")
            break
        print("-"*20)
        
    #exit()
    #torch.save(best_model.state_dict(), "./papw/all_day_repeat20_stage1_5/png/modified/best_model.pth")
    
    
    print("Infering on test set...")
    
    all_data = data["all_data"]
    test_x = all_data[0:2]
    #test_x = np.pad(test_x, ((0, 0), (0, 0), (3, 0), (0, 0)))
    print(test_x.shape)
    
    model = attack_model(node_num=11, device=device, horizon=horizon//2).to(device)
    model.load_state_dict(torch.load("./papw/total_infect/modified/best_attack_model.pth", weights_only=True))
    model.eval()
    model.to(device)
    
    if os.path.exists("./papw/total_infect/modified/graph_mask.txt"):
        os.remove("./papw/total_infect/modified/graph_mask.txt")
    for repeat_idx, test_data_x in tqdm(enumerate(test_x), total=len(test_x)):
        for stage_idx, stage in enumerate(range(test_data_x.shape[0])):
            if stage_idx == 2:
                continue
            data_x = test_data_x[stage_idx]
            modified_graph = graph
            for sld_win in range(0, data_x.shape[0], pred_period):
                
                if sld_win == 0:
                    y = torch.tensor((data_x[0:horizon, 0:11]-mean)/std, dtype=torch.float)
                    y = torch.cat((y, torch.sum(y, dim=-1).unsqueeze(-1)), dim=-1)
                    
                    modified_y = torch.tensor((data_x[0:horizon, 0:11]-mean)/std, dtype=torch.float)
                    modified_y = torch.cat((modified_y, torch.sum(modified_y, dim=-1).unsqueeze(-1)), dim=-1)
                    #print(y[-horizon:, :-1])
                #print(y.shape)
                #import ipdb; ipdb.set_trace()
                
                
                y_idx = surrogate_model.predict(y[-horizon:, 0:11].unsqueeze(0).unsqueeze(-1), graph=graph)
                #y_idx = surrogate_model.predict(torch.tensor(data_x[sld_win:sld_win+horizon, 0:11]/2000, dtype=torch.float).unsqueeze(0).unsqueeze(-1), graph=torch.tensor(graph, dtype=torch.float))
                y_idx = y_idx.squeeze()
                y_idx = torch.cat((y_idx, torch.sum(y_idx, dim=-1).unsqueeze(-1)), dim=-1)
                #print(y_idx.shape)
                if pred_period == 1:
                    y_idx = y_idx.unsqueeze(0)
                y = torch.cat((y, y_idx), dim=0)
                
                modified_graph, graph_mask = model(modified_y[-horizon:, 0:11].to(device), graph, graph_seq)
                modified_y_idx = surrogate_model.predict(modified_y[-horizon:, 0:11].unsqueeze(0).unsqueeze(-1), graph=modified_graph)
                modified_y_idx = modified_y_idx.squeeze()
                modified_y_idx = torch.cat((modified_y_idx, torch.sum(modified_y_idx, dim=-1).unsqueeze(-1)), dim=-1)
                #print(y_idx.shape)
                if pred_period == 1:
                    modified_y_idx = modified_y_idx.unsqueeze(0)
                modified_y = torch.cat((modified_y, modified_y_idx), dim=0)
                
                if stage_idx == 1 and repeat_idx == 0:
                    #print("graph_mask: ", graph_mask)
                    rounded_array = np.round(graph_mask.cpu().detach().squeeze().numpy(), 4)
                    formatted_numbers = ["{:.4f}".format(num) for num in rounded_array]
                    with open("./papw/total_infect/modified/graph_mask.txt","a") as f:
                        f.write("   ".join(formatted_numbers)  + f": {round(y_idx[:, -1].detach().cpu().item()*std+mean*11)} - {round(modified_y_idx[:, -1].detach().cpu().item()*std+mean*11)}\n")
                
                
                #y.append(y_idx.squeeze())
                #print(y.shape)
                
                if y.shape[0] >= all_data.shape[2]:
                    y = y[0:all_data.shape[2]]
                    break
            
            new = np.hstack((y[:, -1, np.newaxis]*std+mean*11, modified_y[:, -1, np.newaxis]*std+mean*11))
            layout = epi.visualize.plot_series(new, columns=['pred_Infect', 'modified_Infect'])
            #plt.show()
            plt.savefig(f'./papw/total_infect/modified/modified_repeat{repeat_idx+1}_stage{stage_idx+1}.png')
            plt.close()
            
            
        
        
