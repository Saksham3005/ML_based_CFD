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
# path = "combined_output_main.csv"
nu = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


path_joined = []
for j in range(0, 21):
    
    path_joined.append(f"Data/Boundary_Data/data_with_boundary{j}.csv")
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

l = []
for i in range(0, 100):
    l.append(i)
l = np.array(l)

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

net = Net().to(device)
criterion = nn.MSELoss()
# optimizer = optim.LBFGS(net.parameters(), lr=0.0001, max_iter=20)
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.05)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=300, gamma=0.1)



max_iters = 1000


def function(x, y, t, b):
        res = net(torch.hstack((x, y, t, b))).to(device)
        #res = self.net(torch.cat(x, y, t), dim = 1)
        # print(res)
        if res.shape[0] != 300:
            u = res[:,torch.arange(0, 300, 3)].to(device)
            v = res[:,torch.arange(1, 300, 3)].to(device)
            p = res[:,torch.arange(2, 300, 3)].to(device)
        else:   
            v = res[torch.arange(1, 300, 3)].to(device)
            p = res[torch.arange(2, 300, 3)].to(device)            
            u = res[torch.arange(0, 300, 3)].to(device)

        # u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0] #retain_graph=True,
        # v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        return u, v, p, f, g
    
for path in path_joined:
    
    PINN_dataset = DATA(path)
    data = DataLoader(PINN_dataset, batch_size=1000, pin_memory=True, num_workers=4)

    def closure():
        optimizer.zero_grad()
        total_loss = 0
        
        # for i, sample in enumerate(data):
        #     # Print shapes of input tensors for debugging
        #     # print(f"\nBatch {i} shapes:")
        #     # for key, value in sample.items():
        #     #     print(f"{key}: {value.shape}")
            
        #     ui = sample['ui'].to(device)
        #     vi = sample['vi'].to(device)
        #     pi = sample['pi'].to(device)
        #     x = sample['x'].float().to(device)
        #     y = sample['y'].float().to(device)
        #     b = sample['bi'].float().to(device)
        #     t = torch.ones((1, 99), dtype=torch.float32).to(device) * (i + 1)
        #     optimizer.zero_grad()
        # total_loss = 0
        
        for i, sample in enumerate(data):
            # Print shapes of input tensors for debugging
            # print(f"\nBatch {i} shapes:")
            # for key, value in sample.items():
            #     print(f"{key}: {value.shape}")
            
            ui = sample['ui'].to(device)
            vi = sample['vi'].to(device)
            pi = sample['pi'].to(device)
            x = sample['x'].float().to(device)
            y = sample['y'].float().to(device)
            b = sample['bi'].float().to(device)
            t = torch.ones((x.shape[0], 100), dtype=torch.float32).to(device) * (i + 1)
            
            x.requires_grad = True
            y.requires_grad = True
            t.requires_grad = True
            b.requires_grad = True

            # Get predictions from function
            U, V, P, F, G = function(x, y, t, b)
            
            # Print shapes after function call
            # print(f"After function call:")
            # print(f"U: {U.shape}, V: {V.shape}, P: {P.shape}")
            # print(f"F: {F.shape}, G: {G.shape}")
            
            # Get target values
            uf = sample['uf'].to(device)
            vf = sample['vf'].to(device)
            pf = sample['pf'].to(device)
            
            # Check if tensors are empty
            if uf.numel() == 0 or U.numel() == 0:
                # print(f"Warning: Empty tensor detected at batch {i}")
                # print(f"uf numel: {uf.numel()}, U numel: {U.numel()}")
                continue

            # Ensure consistent shapes for loss calculation
            # First, ensure both tensors are 2D
            if len(U.shape) == 1:
                U = U.unsqueeze(0)
            if len(uf.shape) == 1:
                uf = uf.unsqueeze(0)
            
            # Ensure they have the same number of dimensions
            while len(U.shape) < len(uf.shape):
                U = U.unsqueeze(0)
            while len(uf.shape) < len(U.shape):
                uf = uf.unsqueeze(0)
                
            # print(f"Before loss calculation:")
            # print(f"U shape: {U.shape}, uf shape: {uf.shape}")
            
            try:
                loss_u = criterion(U, uf)
                loss_v = criterion(V, vf)
                loss_p = criterion(P, pf)
                
                # Get network output for current state
                input = torch.hstack((x, y, t, b)).to(device)
                input = torch.squeeze(input, dim=0).to(device)
                output = net(input)
                
                if output.shape[0] != 300:
                    u = output[:,torch.arange(0, 300, 3)]
                    v = output[:,torch.arange(1, 300, 3)]
                    p = output[:,torch.arange(2, 300, 3)]
                else:
                    u = output[torch.arange(0, 300, 3)]
                    v = output[torch.arange(1, 300, 3)]
                    p = output[torch.arange(2, 300, 3)]

                # Ensure shapes match for next state predictions
                u = u.view_as(uf)
                v = v.view_as(vf)
                p = p.view_as(pf)
                
                loss_u_next = criterion(u, uf)
                loss_v_next = criterion(v, vf)
                loss_p_next = criterion(p, pf)
                
                zero_target = torch.zeros_like(F)
                loss_f = criterion(F, zero_target)
                loss_g = criterion(G, zero_target)

                loss = loss_u + loss_v + loss_p + loss_f + loss_g + loss_u_next + loss_v_next + loss_p_next
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value detected at batch {i}")
                    continue
                    
                total_loss += loss
                
            except RuntimeError as e:
                print(f"Error in batch {i}:")
                print(e)
                print(f"Shapes at error:")
                print(f"U: {U.shape}, uf: {uf.shape}")
                print(f"V: {V.shape}, vf: {vf.shape}")
                print(f"P: {P.shape}, pf: {pf.shape}")
                continue

        if total_loss == 0:
            print("Warning: total_loss is zero. No valid batches processed.")
            return torch.tensor(0.0, requires_grad=True, device=device)
            
        total_loss.backward()
        return total_loss


    running_loss = 0.0
    for epoch in tqdm(range(max_iters)):
        loss = optimizer.step(closure)
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

            
            PATH = 'model_weights_new.pt'
            torch.save(net.state_dict(), PATH)

print('Finished Training')
