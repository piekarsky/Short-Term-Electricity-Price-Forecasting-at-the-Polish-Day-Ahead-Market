import os
import pandas as pd


import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset




def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data):
    """Data Loader"""
    data_dir = os.path.join(data)

    data = pd.read_excel(data_dir)

    #data.index = data['date']
   # data = data.drop('date', axis=1)

    return data

    
def standardization(df):
    
    target = 'value'
    target_mean = df[target].mean()
    target_stdev = df[target].std()

    for c in df.columns:
        mean = df[c].mean()
        stdev = df[c].std()

        df[c] = (df[c] - mean) / stdev
        
    
    return df


class SequenceDataset(Dataset):
    
    display(Dataset)
    
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
           # x = torch.cat((x[:,:i], x[:, i+1:]), axis = 1)
           # x = torch.cat((x[:,:i], x[:, i+1:]))
           # x = x[torch.arange(x.size(0))!=self.sequence_length-1] 
            
            
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    
    print(f"Train loss: {avg_loss}")
  
    
    

def test_model(data_loader, model, loss_function):

    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    
    print(f"Val loss: {avg_loss}")
   
    


