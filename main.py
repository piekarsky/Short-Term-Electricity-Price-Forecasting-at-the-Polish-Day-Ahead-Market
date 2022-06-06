import os
import random
import argparse
import numpy as np
import pandas as pd

from utils import make_dirs, load_data, standardization, SequenceDataset, train_model, test_model, predict
import torch
from models import RNNModel
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader



def main(args):

    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare Data 
    data = load_data(args.which_data)[args.feature]
    
    
    
    df = data.set_index(['date'])
    
       
    #Split Data
    val_start = "2020-07-01 01:00:00"
    test_start = "2020-10-01 01:00:00"

    df_train = df.loc[:val_start].copy()
    df_val = df.loc[val_start:test_start].copy()
    df_test = df.loc[test_start:].copy()
    
    
    
    # Standardize Data 
    df_train = standardization(df_train)
    df_val = standardization(df_val)
    df_test = standardization(df_test)
    
    
    target = "value"
    features = list(df.columns.difference([target]))
      
    


    train_dataset = SequenceDataset(
        df_train,
        target=target,
    
        features=features,
        sequence_length=args.seq_length
    )


    val_dataset = SequenceDataset(
        df_val,
        target=target,
    
        features=features,
        sequence_length=args.seq_length
    )

    test_dataset = SequenceDataset(
        df_test,
        target=target,
    
        features=features,
        sequence_length=args.seq_length
    )
    

    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    if args.model == 'rnn':
        model = RNN(num_inputs=len(features), args.hidden_size, args.num_layers, args.output_size, args.dropout)
    elif args.model == 'lstm':
        model = LSTM(num_inputs=len(features), args.hidden_size, args.num_layers, args.output_size, args.dropout)
    elif args.model == 'gru':
        model = GRU(num_inputs=len(features), args.hidden_size, args.num_layers, args.output_size, args.dropout)
    else:
        raise NotImplementedError
    
         
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    train_plot_losss = []
    val_plot_losss = []

    for ix_epoch in range(args.num_epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
       # train_plot_losss.append(train_model(train_loader, model, loss_function, optimizer=optimizer))
        test_model(val_loader, model, loss_function)
  
        print()
    


    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_val[ystar_col] = predict(val_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_val, df_test))[[target, ystar_col]]

    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean

    df_out = df_out.reset_index()
    df_out = df_out[df_out['date']>='2020-10-01 01:00:00']


    result_metrics = calculate_metrics(df_out)
    
  
   
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--which_data', type=str, default='./data/data.xlsx', help='which data to use')
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')   
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--feature', type=str, 
                         default=['value', 'date', 
                                  'electricity demand',
                                  'generation of energy from wind sources', 
                                  'is_weekend',
                                  'code of the day', 
                                  'value lag24', 
                                  'value lag48', 
                                  'value lag72',
                                  'value lag96',
                                  'value lag120', 
                                  'value lag144', 
                                  'value lag168',
                                  'value lag336', 
                                  'electricity demand lag24', 
                                  'electricity demand lag48',
                                  'electricity demand lag72', 
                                  'electricity demand lag96',
                                  'electricity demand lag120', 
                                  'electricity demand lag144',
                                  'electricity demand lag168', 
                                  'electricity demand lag336',
                                  'generation of energy from wind sources lag24',
                                  'generation of energy from wind sources lag48',
                                  'generation of energy from wind sources lag72',
                                  'generation of energy from wind sources lag96',
                                  'generation of energy from wind sources lag120',
                                  'generation of energy from wind sources lag144',
                                  'generation of energy from wind sources lag168',
                                  'generation of energy from wind sources lag336'], help='extract which feature')
    
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--num_hidden_units', type=int, default=256, help='hidden units')
    parser.add_argument('--num_epochs', type=int, default=10, help='num epochs')
    parser.add_argument('--num_layers', type=int, default=10, help='num layer dim')
    parser.add_argument('--dropout', type=int, default=10, help='dropout rate')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru'])
    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)

