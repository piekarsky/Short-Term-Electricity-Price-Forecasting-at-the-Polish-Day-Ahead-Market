#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(data):
    
    data_dir = os.path.join(data)
    data = pd.read_excel(data_dir)
    data.index = data['date']
    data = data.drop('date', axis=1)

    return data


def plot_full(path, data, feature):
    """Plot Full Graph of Energy Dataset"""
    data.plot(y=feature, figsize=(16, 8))
    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel(feature, fontsize=10)
    plt.grid()
    plt.title('{} Energy Prediction'.format(feature))
    plt.savefig(os.path.join(path, '{} Energy Prediction.png'.format(feature)))
    plt.show()


# In[15]:





# In[ ]:




