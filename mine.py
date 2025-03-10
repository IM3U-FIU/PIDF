import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class mine_net(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = self.fc2(output)
        return output

class mine:
    def __init__(self, p_dis, q_dis, num_iterations, all = True, batch_size = 5000,lr = 0.001):
        self.lr = lr
        self.all = all
        self.ma_window_size = int(num_iterations/5)
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.obs = p_dis.round(decimals =2)
        self.acs = q_dis.round(decimals =2)
        if len(self.obs.shape) == 1:
            self.obs = np.expand_dims(self.obs, 1)
        if len(self.acs.shape) == 1:
            self.acs = np.expand_dims(self.acs, 1)
        self.expts =3

    def kullback_liebler(self, dis_p, dis_q, kl_net):
        t = kl_net(dis_p)
        et = torch.exp(kl_net(dis_q))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et)) #- (0.01*2**torch.log(torch.mean(et))*self.obs.shape[1])
        return mi_lb, t, et

    def learn_klne(self, batch, mine_net, mine_net_optim, ma_et, ma_rate=0.001):
        joint, marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
        mi_lb, t, et = self.kullback_liebler(joint, marginal, mine_net)
        ma_et = (1-ma_rate) * ma_et + ma_rate * torch.mean(et)
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach() * torch.mean(et))
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et
   
    def trip_sample_batch(self, sample_mode='joint'):
        index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
        if sample_mode == 'marginal':
            marginal_index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
            
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[marginal_index, :])),axis=1)
        else:
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[ index, :])),axis=1)
        
        return batch

    def trip_train(self, tripmine_net, tripmine_net_optim):
        ma_et = 1.
        result = list()
        for i in range(self.num_iterations):
            batch = self.trip_sample_batch(), self.trip_sample_batch(sample_mode='marginal') 
            mi_lb, ma_et = self.learn_klne(batch, tripmine_net, tripmine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
        return result

    def ma(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]

    def trip_initialiser(self):
        tripmine_net = mine_net(self.obs.shape[1]+self.acs.shape[1]).to(device)
        tripmine_net_optim = optim.Adam(tripmine_net.parameters(), lr=self.lr)
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append(self.ma(self.trip_train( tripmine_net, tripmine_net_optim)))
        return np.array(trip_results)

    
    def run(self):
        results = self.trip_initialiser()
        return results

    
class mine_fa:
    def __init__(self, p_dis, q_dis, num_iterations, all=True, batch_size=5000, lr=0.001):
        self.lr = lr
        self.all = all
        self.ma_window_size =  int(num_iterations / 5)
        # Convert p_dis and q_dis to PyTorch tensors if they are not already
        if not isinstance(p_dis, torch.Tensor):
            p_dis = torch.tensor(p_dis, dtype=torch.float32)
        if not isinstance(q_dis, torch.Tensor):
            q_dis = torch.tensor(q_dis, dtype=torch.float32)
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size  
        self.num_iterations = num_iterations
        # Round the tensors to 2 decimal places
        self.p_dis = p_dis.round()
        self.q_dis = q_dis.round()
        # Ensure p_dis and q_dis are 2D (N x 1 if they are initially 1D)
        if len(self.p_dis.shape) == 1:
            self.p_dis = self.p_dis.unsqueeze(1)
        elif self.p_dis.shape[1] == 0:
            N = self.p_dis.shape[0]
            self.p_dis = torch.zeros(N, 1)
        if len(self.q_dis.shape) == 1:
            self.q_dis = self.q_dis.unsqueeze(1) 
        elif self.q_dis.shape[1] == 0:
            N = self.q_dis.shape[0]
            self.q_dis = torch.zeros(N, 1)  
        self.expts = 3

    def fit_mlp(self):
        mlp_regressor = nn.Sequential(
            nn.Linear(self.p_dis.shape[1], 50), nn.ReLU(),nn.Linear(50, self.q_dis.shape[1])
        ).to(device) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=0.001)
        losses_feats = []
        for epoch in range(self.num_iterations):
                
            # Reset gradients
            optimizer.zero_grad()
                
            # Generate predictions for the entire dataset
            predictions = mlp_regressor(self.p_dis.to(device))
            loss = criterion(predictions, self.q_dis.to(device))
            loss.backward()
            optimizer.step()
            losses_feats.append(loss.item())

        losses_no_feats = []
        mlp_regressor = nn.Sequential(
            nn.Linear(self.p_dis.shape[1], 50), nn.ReLU(),nn.Linear(50, self.q_dis.shape[1])
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=0.001)
        for epoch in range(self.num_iterations):
                
            # Reset gradients
            optimizer.zero_grad()
                
            # Generate predictions for the entire dataset
            predictions = mlp_regressor(torch.zeros_like(self.p_dis).to(device))
            loss = criterion(predictions, self.q_dis.to(device))
            loss.backward()
            optimizer.step()
            losses_no_feats.append(loss.item())

        return np.array(losses_no_feats) - np.array(losses_feats)
    
    def ma(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]
    
    def run(self):
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append(self.ma(self.fit_mlp()))
        return np.array(trip_results)
    
class mine_fa_cel:
    def __init__(self, p_dis, q_dis, num_iterations, all=True, batch_size=5000, lr=0.001):
        self.lr = lr
        self.all = all
        self.ma_window_size = int(num_iterations / 5)
        # Convert p_dis and q_dis to PyTorch tensors if they are not already
        if not isinstance(p_dis, torch.Tensor):
            p_dis = torch.tensor(p_dis, dtype=torch.float32)
        if not isinstance(q_dis, torch.Tensor):
            q_dis = torch.tensor(q_dis, dtype=torch.long)  # Assume q_dis is already in the form of class indices
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.p_dis = p_dis
        self.q_dis = q_dis.squeeze()  # Ensure q_dis is a 1D tensor
        # Ensure p_dis is 2D (N x 1 if initially 1D)
        if len(self.p_dis.shape) == 1:
            self.p_dis = self.p_dis.unsqueeze(1)
        self.expts = 3

    def fit_mlp(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mlp_regressor = nn.Sequential(
            nn.Linear(self.p_dis.shape[1], 50), nn.ReLU(),
            nn.Linear(50, self.q_dis.max().item()+1)  # Adjust output layer to match number of classes
        ).to(device) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp_regressor.parameters(), lr=self.lr)
        losses_feats = []
        for epoch in range(self.num_iterations):
            optimizer.zero_grad()
            predictions = mlp_regressor(self.p_dis.to(device))
            loss = criterion(predictions, self.q_dis.to(device))
            loss.backward()
            optimizer.step()
            losses_feats.append(loss.item())

        # Similar loop for the zero-feature scenario, adjust if necessary for your specific needs
        mlp_regressor = nn.Sequential(
            nn.Linear(self.p_dis.shape[1], 50), nn.ReLU(),
            nn.Linear(50, self.q_dis.max().item()+1)  # Adjust output layer to match number of classes
        ).to(device) 
        losses_no_feats = []
        for epoch in range(self.num_iterations):
            optimizer.zero_grad()
            predictions = mlp_regressor(torch.zeros_like(self.p_dis).to(device))
            loss = criterion(predictions, self.q_dis.to(device))
            loss.backward()
            optimizer.step()
            losses_no_feats.append(loss.item())

        return np.array(losses_no_feats) - np.array(losses_feats)
    
    def ma(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]
    
    def run(self):
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append(self.ma(self.fit_mlp()))
        return np.array(trip_results)

