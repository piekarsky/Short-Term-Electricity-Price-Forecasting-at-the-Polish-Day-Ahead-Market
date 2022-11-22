import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from utils import make_dirs, load_data, standardization, train_validate_test_split, SequenceDataset, train_model, val_model, predict
from models import RNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader



def main(args):

    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare Data 
    df = load_data(args.which_data)[args.feature]
    df = df.set_index(['date'])  
    features = list(df.columns.difference([args.target]))
    target_mean = df[args.target].mean()
    target_stdev = df[args.target].std()
    
    # Standardize Data 
    df_ = standardization(df, args.target)
    df = df_.reset_index(drop=False)
    
    
    #Split Data
    df_train, df_val, df_test = train_validate_test_split(df, args.test_split) 
    df_train = df_train.set_index(['date'])
    df_val = df_val.set_index(['date'])
    df_test = df_test.set_index(['date'])
    
    display(df_train)
    
    train_dataset = SequenceDataset(
        df_train,
        target=args.target,    
        features=features,
        sequence_length=args.seq_length
    )


    val_dataset = SequenceDataset(
        df_val,
        target=args.target,  
        features=features,
        sequence_length=args.seq_length
    )

    test_dataset = SequenceDataset(
        df_test,
        target=args.target,   
        features=features,
        sequence_length=args.seq_length
    )
     

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_inputs=len(features)
    
    if args.model == 'rnn':
        model = RNN(num_inputs, args.num_hidden_size, args.num_layers, args.output_size, args.dropout)
  #  elif args.model == 'lstm':
    #    model = LSTM(num_inputs=len(features), args.hidden_size, args.num_layers, args.output_size, args.dropout)
   # elif args.model == 'gru':
     #   model = GRU(num_inputs=len(features), args.hidden_size, args.num_layers, args.output_size, args.dropout)
    else:
        raise NotImplementedError
    
         
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_plot_losss = []
    val_plot_losss = []

    for ix_epoch in range(1, args.num_epochs+1):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        val_model(val_loader, model, loss_function)
        print()
    
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    ystar_col = "model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_val[ystar_col] = predict(val_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()
    df_out = pd.concat((df_train, df_val, df_test))[[args.target, ystar_col]]

    for c in df_out.columns:
         df_out[c] = df_out[c] * target_stdev + target_mean
    
    display(df_out)

#     df_out = df_out.reset_index()
    

    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_data', type=str, default='./data/data.xlsx', help='which data to use')
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--target', type=str, default='value', help='explained variable')
    parser.add_argument('--output_size', type=int, default=1, help='output_dim')   
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
                                  'generation of energy from wind sources lag336'], help='features')
    parser.add_argument('--test_split', type=float, default=0.2, help='test_split')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--num_hidden_size', type=int, default=256, help='hidden units')
    parser.add_argument('--num_epochs', type=int, default=3, help='num epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='num layer dim')
    parser.add_argument('--dropout', type=int, default=0.6, help='dropout rate')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru'])
    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)

