import torch
import torch.nn as nn
import math

class RNN(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()
        self.num_features = num_features 
        
        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units
        self.num_layers = num_layers       
        self.output_size = output_size
        self.dropout = dropout_rate

        # RNN layers
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate           
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_size)

    def forward(self, x):
        
        batch_size = x.shape[0]       
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()       
        _, hn = self.rnn(x, h0)
                
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out


class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super().__init__()
        self.num_features = num_features  # 
        
        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units       
        self.num_layers = num_layers       
        self.output_size = output_size
        self.dropout = dropout_rate

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate
        )
        # Fully connected layer
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above
        return out
    
    
class GRU(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        self.num_features = num_features  
        
        # Defining the number of layers and the nodes in each layer
        self.hidden_units = hidden_units       
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout_rate

        # GRU layers
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout_rate
        )       
        # Fully connected layer
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_size)

    def forward(self, x):
        
        batch_size = x.shape[0]
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())         
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.      
        return out

   