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

    data.index = data['date']
    data = data.drop('date', axis=1)

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
          
            
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    """Rolling Window Function for Multi-step"""
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)[:, :, 0]



def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape
