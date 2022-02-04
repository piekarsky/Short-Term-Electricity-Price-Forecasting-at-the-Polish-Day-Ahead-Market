#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch


from utils import make_dirs, load_data


def main(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    
    paths = [args.weights_path, args.plots_path, args.numpy_path]
    for path in paths:
        make_dirs(path)
        
    data = load_data(args.which_data)[[args.feature]]
    data = data.copy()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_data', type=str, default='./data/energydata_complete.csv', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')
    parser.add_argument('--numpy_path', type=str, default='./results/numpy/', help='numpy path')
    parser.add_argument('--feature', type=str, default='Appliances', help='extract which feature for prediction')
    
    config = parser.parse_args()
    
    main(config)


# In[ ]:




