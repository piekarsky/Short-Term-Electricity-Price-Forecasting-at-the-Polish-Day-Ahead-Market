import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from utils import make_dirs, load_data, standardization, train_validate_test_split, SequenceDataset, train_model, val_model, predict, inverse_transform, calculate_metrics
from models import RNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader



def main(args):

    # Fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare data 
    df = load_data(args.which_data)[args.feature]
    df = df.set_index(['date'])  
    features = list(df.columns.difference([args.target]))
    target_mean = df[args.target].mean()
    target_stdev = df[args.target].std()
    
    # Standardize data 
    df_ = standardization(df, args.target)
    df = df_.reset_index(drop=False)
    
    
    #Split data
    df_train, df_val, df_test = train_validate_test_split(df, args.test_split) 
    df_train = df_train.set_index(['date'])
    df_val = df_val.set_index(['date'])
    df_test = df_test.set_index(['date'])
    
    display(df_train)
    display(df_val)
    display(df_test)
    
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
    elif args.model == 'lstm':
        model = LSTM(num_inputs, args.hidden_size, args.num_layers, args.output_size, args.dropout)
    elif args.model == 'gru':
        model = GRU(num_inputs, args.hidden_size, args.num_layers, args.output_size, args.dropout)
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
    
    
#    def inverse_transform(scaler, df, columns):
#    for col in columns:
#        df[col] = scaler.inverse_transform(df[col])
#    return df
    df_pred = inverse_transform(df_out, target_stdev,  target_mean)
    result_metrics = calculate_metrics(df_pred, args.target, ystar_col)
    display(result_metrics)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_data', type=str, default='./data/data.xlsx', help='which data to use')
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')   
    parser.add_argument('--output_size', type=int, default=1, help='output_dim')   
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--target', type=str, default='fixing_I_course (PLN/MWh)', help='explained variable')
    parser.add_argument('--feature', type=str, 
                         default=['date', 
                                  'fixing_I_course (PLN/MWh)',
                                  'domestic_electricity_demand (MW)',
                                  'generation_of_energy_from_wind_sources (MW)', 
                                  'is_holiday',
                                  'code_of_the_day', 
                                  'fixing_I_course (PLN/MWh) lag24', 
                                  'fixing_I_course (PLN/MWh) lag48', 
                                  'fixing_I_course (PLN/MWh) lag72',
                                  'fixing_I_course (PLN/MWh) lag96',
                                  'fixing_I_course (PLN/MWh) lag120', 
                                  'fixing_I_course (PLN/MWh) lag144', 
                                  'fixing_I_course (PLN/MWh) lag168',
                                  'fixing_I_course (PLN/MWh) lag336', 
                                  'domestic_electricity_demand (MW) lag24', 
                                  'domestic_electricity_demand (MW) lag48',
                                  'domestic_electricity_demand (MW) lag72', 
                                  'domestic_electricity_demand (MW) lag96',
                                  'domestic_electricity_demand (MW) lag120', 
                                  'domestic_electricity_demand (MW) lag144',
                                  'domestic_electricity_demand (MW) lag168', 
                                  'domestic_electricity_demand (MW) lag336',
                                  'generation_of_energy_from_wind_sources (MW) lag24',
                                  'generation_of_energy_from_wind_sources (MW) lag48',
                                  'generation_of_energy_from_wind_sources (MW) lag72',
                                  'generation_of_energy_from_wind_sources (MW) lag96',
                                  'generation_of_energy_from_wind_sources (MW) lag120',
                                  'generation_of_energy_from_wind_sources (MW) lag144',
                                  'generation_of_energy_from_wind_sources (MW) lag168',
                                  'generation_of_energy_from_wind_sources (MW) lag336'], help='ex_features')
    parser.add_argument('--test_split', type=float, default=0.0824, help='test_split')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--num_hidden_size', type=int, default=256, help='hidden units')
    parser.add_argument('--num_epochs', type=int, default=11, help='num epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='num layer dim')
    parser.add_argument('--dropout', type=int, default=0.6, help='dropout rate')
    parser.add_argument('--model', type=str, default='rnn', choices=['rnn', 'lstm', 'gru'])
    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)

