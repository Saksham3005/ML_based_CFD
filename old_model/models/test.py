import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Testing Code
test_path = "Boundary_Data/combined_output_main.csv"  # Path to the test dataset

# Define the test dataset class
class TestDATA(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        df = pd.read_csv(path)

        self.u_min, self.u_max = df['u'].min(), df['u'].max()
        self.v_min, self.v_max = df['v'].min(), df['v'].max()
        self.p_min, self.p_max = df['p'].min(), df['p'].max()
        self.df = df

    def normalize(self, values, min_val, max_val):
        return (max_val - values) / (max_val - min_val)

    def __len__(self):
        return int(len(self.df) / 100) - 1

    def __getitem__(self, idx):
        ui = self.df.iloc[idx * 100 : (1 + idx) * 100]['u']
        vi = self.df.iloc[idx * 100 : (1 + idx) * 100]['v']
        pi = self.df.iloc[idx * 100 : (1 + idx) * 100]['p']
        bi = self.df.iloc[idx * 100 : (1 + idx) * 100]['Boundary']

        uf = self.df.iloc[(idx + 1) * 100 : (2 + idx) * 100]['u']
        vf = self.df.iloc[(idx + 1) * 100 : (2 + idx) * 100]['v']
        pf = self.df.iloc[(idx + 1) * 100 : (2 + idx) * 100]['p']

        # Apply min-max normalization
        ui = self.normalize(ui, self.u_min, self.u_max)
        vi = self.normalize(vi, self.v_min, self.v_max)
        pi = self.normalize(pi, self.p_min, self.p_max)

        uf = self.normalize(uf, self.u_min, self.u_max)
        vf = self.normalize(vf, self.v_min, self.v_max)
        pf = self.normalize(pf, self.p_min, self.p_max)

        sample = {
            'ui': torch.tensor(ui.to_numpy(), dtype=torch.float32),
            'vi': torch.tensor(vi.to_numpy(), dtype=torch.float32),
            'pi': torch.tensor(pi.to_numpy(), dtype=torch.float32),
            'bi': torch.tensor(bi.to_numpy(), dtype=torch.int8),
            'uf': torch.tensor(uf.to_numpy(), dtype=torch.float32),
            'vf': torch.tensor(vf.to_numpy(), dtype=torch.float32),
            'pf': torch.tensor(pf.to_numpy(), dtype=torch.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
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
    
    
net = Net().to(device)
criterion = nn.MSELoss()


PINN_dataset = TestDATA(test_path)





test_loader = DataLoader(PINN_dataset, batch_size=1, pin_memory=True, num_workers=4)

net = Net().to(device)
net.load_state_dict(torch.load('best_params/model_new.pt'))
net.eval() \

test_loss = 0.0
criterion = nn.MSELoss()

threshold = 0.05  


total_predictions = 0
correct_predictions = 0


test_loss = 0.0

with torch.no_grad(): 
    for i, sample in enumerate(test_loader):
        ui = sample['ui'].to(device)
        vi = sample['vi'].to(device)
        pi = sample['pi'].to(device)
        bi = sample['bi'].to(device)

        uf = sample['uf'].to(device)
        vf = sample['vf'].to(device)
        pf = sample['pf'].to(device)

        input = torch.hstack((ui, vi, pi, bi))
        input = torch.squeeze(input, dim=0).to(device)
        output = net(input)

        if output.shape[0] != 300:
            u = output[:, torch.arange(0, 300, 3)].to(device)
            v = output[:, torch.arange(1, 300, 3)].to(device)
            p = output[:, torch.arange(2, 300, 3)].to(device)
        else:
            v = output[torch.arange(1, 300, 3)].to(device)
            p = output[torch.arange(2, 300, 3)].to(device)
            u = output[torch.arange(0, 300, 3)].to(device)

        loss_u = criterion(uf, u)
        loss_v = criterion(vf, v)
        loss_p = criterion(pf, p)

        loss = loss_u + loss_v + loss_p
        test_loss += loss.item()


        total_predictions += uf.numel() + vf.numel() + pf.numel()
        correct_predictions += ((torch.abs(uf - u) < threshold).sum().item() +
                                (torch.abs(vf - v) < threshold).sum().item() +
                                (torch.abs(pf - p) < threshold).sum().item())


        print(f"Sample {i}:")
        print(f"Predicted u: {u}")
        print(f"Ground truth u: {uf}")

average_test_loss = test_loss / len(test_loader)
accuracy = (correct_predictions / total_predictions) * 100

print(f"Average Test Loss: {average_test_loss}")
print(f"Accuracy: {accuracy:.2f}%")