import os
import random
import argparse
import numpy as np
import pandas as pd

from utils import make_dirs, load_data, standardization, SequenceDataset
import torch
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
    #print(data)
    
    
    df = data.set_index(['date'])
    
    #display(df)
    
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--which_data', type=str, default='./data/data.xlsx', help='which data to use')
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--feature', type=str, default=['value', 'date', 'value lag24', 'value lag48', 'value lag72', 'value lag96', 
                                                        'value lag120', 'value lag144'], help='extract which feature')
    
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    
    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)

