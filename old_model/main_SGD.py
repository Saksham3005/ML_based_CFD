import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import torch.optim as optim


from tqdm import tqdm


##constants
path = "Boundary_Data/data_with_boundary4.csv"
nu = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(400, 390)   
        self.fc2 = nn.Linear(390, 380)  
        self.fc3 = nn.Linear(380, 370)  
        self.fc4 = nn.Linear(370, 360)  
        self.fc5 = nn.Linear(360, 350)  
        self.fc6 = nn.Linear(350, 340)  
        self.fc7 = nn.Linear(340, 330)  
        self.fc8 = nn.Linear(330, 320)  
        self.fc9 = nn.Linear(320, 310)  
        self.fc10 = nn.Linear(310, 300)  

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
  
        x = self.fc10(x)              
        return x


class DATA(Dataset):

    def __init__(self, path, transform= None):
        self.path = path
        self.transform = transform
        df = pd.read_csv(path)        
        self.df = df

    def __len__(self):
        return int(len(self.df)/100) - 1

    def __getitem__(self, idx):
        
        ui = self.df.iloc[idx*100 : (1+idx)*100]['u']
        vi = self.df.iloc[idx*100 : (1+idx)*100]['v']
        pi = self.df.iloc[idx*100 : (1+idx)*100]['p']
        bi = self.df.iloc[idx*100 : (1+idx)*100]['Boundary']

        i = idx+1

        uf = self.df.iloc[i*100 : (1+i)*100]['u']
        vf = self.df.iloc[i*100 : (1+i)*100]['v']
        pf = self.df.iloc[i*100 : (1+i)*100]['p']
        bf = self.df.iloc[i*100 : (1+i)*100]['Boundary']

        sample = {

            'ui': torch.tensor(ui.to_numpy(), dtype=torch.float32),
            'vi': torch.tensor(vi.to_numpy(), dtype=torch.float32),
            'pi': torch.tensor(pi.to_numpy(), dtype=torch.float32),
            'bi': torch.tensor(bi.to_numpy(), dtype=torch.int8),

            'uf': torch.tensor(uf.to_numpy(), dtype=torch.float32),
            'vf': torch.tensor(vf.to_numpy(), dtype=torch.float32),
            'pf': torch.tensor(pf.to_numpy(), dtype=torch.float32),
            'bf': torch.tensor(bf.to_numpy(), dtype=torch.int8)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


net = Net().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2)

PINN_dataset = DATA(path)
data = DataLoader(PINN_dataset, batch_size=1)


# for i, sample in enumerate(data):
#     print(sample)
#     print(i)
#     if i == 999:
#         break
max_iters = 1000

for epochs in tqdm(range(0, max_iters)):
    running_loss = 0.0
    
    
    for i, sample in enumerate(data):

        optimizer.zero_grad()
        
        ui = sample['ui'].to(device)
        vi = sample['vi'].to(device)
        pi = sample['pi'].to(device)
        bi = sample['bi'].to(device)

        uf = sample['uf'].to(device)
        vf = sample['vf'].to(device)
        pf = sample['pf'].to(device)
        bf = sample['bf'].to(device)


        input = torch.hstack((ui, vi, pi, bi))
        input = torch.squeeze(input, dim = 0)
        # print(input.shape)
        output = net(input)
        
        u = output[torch.arange(0, 300, 3)]
        v = output[torch.arange(1, 300, 3)]
        p = output[torch.arange(2, 300, 3)]
        
        # print(output)
        
        loss_u = criterion(uf, u)
        loss_v = criterion(vf, v)
        loss_p = criterion(pf, p)

        loss = loss_u + loss_v + loss_p

        loss.backward() 
        optimizer.step()
        

        running_loss += loss.item()

        # print("epoch number: ", epoch, "datapoint: ", i, "loss: ", running_loss)
        
        
    # running_loss = 0.0
    if epochs%5 == 0:
        print(running_loss)
        
    running_loss = 0.0
    
    
    
print('Finished Training')

PATH = 'model.pt'
torch.save(net.state_dict(), PATH)