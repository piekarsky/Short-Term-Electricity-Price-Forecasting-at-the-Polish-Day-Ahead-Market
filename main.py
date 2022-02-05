import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from models import DNN, RNN, LSTM, GRU, AttentionalLSTM, CNN
from utils import make_dirs, load_data, plot_full, data_loader, split_sequence_uni_step, split_sequence_multi_step
from utils import get_lr_scheduler, mean_percentage_error, mean_absolute_percentage_error, plot_pred_test

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def main(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    paths = [args.weights_path, args.plots_path, args.numpy_path]
    for path in paths:
        make_dirs(path)
        
    data = load_data(args.which_data)[args.feature]
    data = data.copy()
    
    print(data)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--feature', type=str, default=["lag24", "lag48"], help='extract which feature for prediction')
    parser.add_argument('--which_data', type=str, default='./data/data.xlsx', help='which data to use')
    
    parser.add_argument('--multi_step', type=bool, default=False, help='multi-step or not')
    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    

    config = parser.parse_args()
    
    
    
    main(config)

    



