import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# Constants
path = []
for j in range(0, 21):
    
    path.append(f"Boundary_Data/data_with_boundary{j}.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

l = []
for i in range(0, 100):
    l.append(i)
l = np.array(l)
# Define the model architecture
class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(400, 395)   
        self.fc2 = nn.Linear(395, 390)  
        self.fc3 = nn.Linear(390, 385)  
        self.fc4 = nn.Linear(385, 380)  
        self.fc5 = nn.Linear(380, 375)  
        self.fc6 = nn.Linear(375, 370)  
        self.fc7 = nn.Linear(370, 365)  
        self.fc8 = nn.Linear(365, 360)  
        self.fc9 = nn.Linear(360, 355)  
        self.fc10 = nn.Linear(355, 350)
        self.fc11 = nn.Linear(350, 345)
        self.fc12 = nn.Linear(345, 340)
        self.fc13 = nn.Linear(340, 335)
        self.fc14 = nn.Linear(335, 330)
        self.fc15 = nn.Linear(330, 325)
        self.fc16 = nn.Linear(325, 320)
        self.fc17 = nn.Linear(320, 315)
        self.fc18 = nn.Linear(315, 310)
        self.fc19 = nn.Linear(310, 305)
        self.fc20 = nn.Linear(305, 300)
        # self.fc21 = nn.Linear(305, 305)
        # self.fc22 = nn.Linear(305, 305)
        # self.fc23 = nn.Linear(305, 305)
        # self.fc24 = nn.Linear(305, 305)
        # self.fc25 = nn.Linear(305, 305)
        # self.fc26 = nn.Linear(305, 305)
        # self.fc27 = nn.Linear(305, 305)
        # self.fc28 = nn.Linear(305, 305)
        # self.fc29 = nn.Linear(305, 305)
        # self.fc30 = nn.Linear(305, 300)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = torch.tanh(self.fc4(x)) 
        x = torch.tanh(self.fc5(x)) 
        x = torch.tanh(self.fc6(x)) 
        x = torch.tanh(self.fc7(x)) 
        x = torch.tanh(self.fc8(x)) 
        x = torch.tanh(self.fc9(x))
        x = torch.tanh(self.fc10(x)) 
        x = torch.tanh(self.fc11(x)) 
        x = torch.tanh(self.fc12(x)) 
        x = torch.tanh(self.fc13(x)) 
        x = torch.tanh(self.fc14(x))
        x = torch.tanh(self.fc15(x)) 
        x = torch.tanh(self.fc16(x)) 
        x = torch.tanh(self.fc17(x)) 
        x = torch.tanh(self.fc18(x)) 
        x = torch.tanh(self.fc19(x)) 
        # x = torch.tanh(self.fc20(x)) 
        # x = torch.tanh(self.fc21(x)) 
        # x = torch.tanh(self.fc22(x)) 
        # x = torch.tanh(self.fc23(x)) 
        # x = torch.tanh(self.fc24(x)) 
        # x = torch.tanh(self.fc25(x)) 
        # x = torch.tanh(self.fc26(x)) 
        # x = torch.tanh(self.fc27(x)) 
        # x = torch.tanh(self.fc28(x)) 
        # x = torch.tanh(self.fc29(x)) 
        x = self.fc20(x)              
        return x

class DATA(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        df = pd.read_csv(paths)

    #     # Compute min and max for normalization
    #     self.u_min, self.u_max = df['u'].min(), df['u'].max()
    #     self.v_min, self.v_max = df['v'].min(), df['v'].max()
    #     self.p_min, self.p_max = df['p'].min(), df['p'].max()
        
        self.df = df

    # def normalize(self, values, min_val, max_val):
    #     return (values - min_val) / (max_val - min_val)

    def __len__(self):
        return int(len(self.df)/100) - 1

    def __getitem__(self, idx):
        ui = self.df.iloc[idx*100 : (1+idx)*100]['u']
        vi = self.df.iloc[idx*100 : (1+idx)*100]['v']
        pi = self.df.iloc[idx*100 : (1+idx)*100]['p']
        bi = self.df.iloc[idx*100 : (1+idx)*100]['Boundary']
        x = self.df.iloc[idx*100 : (1+idx)*100]['x']
        y = self.df.iloc[idx*100 : (1+idx)*100]['y']

        i = idx+1

        uf = self.df.iloc[i*100 : (1+i)*100]['u']
        vf = self.df.iloc[i*100 : (1+i)*100]['v']
        pf = self.df.iloc[i*100 : (1+i)*100]['p']
        bf = self.df.iloc[i*100 : (1+i)*100]['Boundary']

        # Apply min-max normalization
        # ui = self.normalize(ui, self.u_min, self.u_max)
        # vi = self.normalize(vi, self.v_min, self.v_max)
        # pi = self.normalize(pi, self.p_min, self.p_max)

        # uf = self.normalize(uf, self.u_min, self.u_max)
        # vf = self.normalize(vf, self.v_min, self.v_max)
        # pf = self.normalize(pf, self.p_min, self.p_max)

        sample = {
            'ui': torch.tensor(ui.to_numpy(), dtype=torch.float32),
            'vi': torch.tensor(vi.to_numpy(), dtype=torch.float32),
            'pi': torch.tensor(pi.to_numpy(), dtype=torch.float32),
            'bi': torch.tensor(bi.to_numpy(), dtype=torch.float32),
            'uf': torch.tensor(uf.to_numpy(), dtype=torch.float32),
            'vf': torch.tensor(vf.to_numpy(), dtype=torch.float32),
            'pf': torch.tensor(pf.to_numpy(), dtype=torch.float32),
            'bf': torch.tensor(bf.to_numpy(), dtype=torch.float32),
            'x' : torch.tensor(x.to_numpy(), dtype=torch.float32),
            'y' : torch.tensor(y.to_numpy(), dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Load the pretrained model
model = Net().to(device)
model.load_state_dict(torch.load('model_weights_new.pt'))
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
k = 0
for pt in path:
    
    PINN_dataset = DATA(pt)
    data = DataLoader(PINN_dataset, batch_size=1, pin_memory=True, num_workers=4)

    with torch.no_grad():
            
        for i, sample in enumerate(data):
            
                ui = sample['ui'].to(device)
                vi = sample['vi'].to(device)
                pi = sample['pi'].to(device)
                bi = sample['bi'].to(device)
                
                u_min, u_max = ui.min(), ui.max()
                v_min, v_max = vi.min(), vi.max()
                p_min, p_max = pi.min(), pi.max()
                
                input = torch.hstack((ui, vi, pi, bi))
                input = torch.squeeze(input, dim=0).to(device)
            
                output = model(input)

                # Extract u, v, p from output
                # u_pred = output[0, torch.arange(0, 300, 3)].cpu().numpy()
                # v_pred = output[0, torch.arange(1, 300, 3)].cpu().numpy()
                # p_pred = output[0, torch.arange(2, 300, 3)].cpu().numpy()
                
                if output.shape[0] != 300:
                    u = output[:, torch.arange(0, 300, 3)].cpu().numpy()
                    v = output[:, torch.arange(1, 300, 3)].cpu().numpy()
                    p = output[:, torch.arange(2, 300, 3)].cpu().numpy()
                else:
                    v = output[torch.arange(1, 300, 3)].cpu().numpy()
                    p = output[torch.arange(2, 300, 3)].cpu().numpy()
                    u = output[torch.arange(0, 300, 3)].cpu().numpy()

            # Denormalize predictions
                # Denormalize predictions
                u_pred = abs(-u * (u_max.cpu().numpy() - u_min.cpu().numpy()) + u_max.cpu().numpy())
                v_pred = abs(-v * (v_max.cpu().numpy() - v_min.cpu().numpy()) + v_max.cpu().numpy())
                p_pred = abs(-p * (p_max.cpu().numpy() - p_min.cpu().numpy()) + p_max.cpu().numpy())

                
                u_pred = -u * (u_max.cpu().numpy() - u_min.cpu().numpy()) + u_max.cpu().numpy()
                v_pred = -v * (v_max.cpu().numpy() - v_min.cpu().numpy()) + v_max.cpu().numpy()
                p_pred = -p * (p_max.cpu().numpy() - p_min.cpu().numpy()) + p_max.cpu().numpy()

                # u_pred = u 
                # v_pred = v 
                # p_pred = p 
                
                # Save predictions
                predictions = pd.DataFrame({
                    'u_pred': u_pred,
                    'v_pred': v_pred,
                    'p_pred': p_pred.flatten(),
                    'x' : np.ones(100, dtype = int)*i,
                    'y' : l.flatten()
                })
                predictions.to_csv(f'Final_pred/predictions_{k}_{i}.csv', index=False)
    k += 1
print("Predictions saved to individual CSV files")