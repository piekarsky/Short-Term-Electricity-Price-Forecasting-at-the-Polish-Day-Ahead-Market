# Short-Term-Electricity-Price-Forecasting-at-the-Polish-Power-Exchange-Day-Ahead-Market


### 1. Overview
This repository contains the experimental source code for short term electricity price forecasting at the Polish Power Exchange Day Ahead Market  including RNN, LSTM, GRU, MLP and Prophet models.

According to the table below, **RNN** outperformed the other models

| Model | MAE↓ | RMSE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|
| Prophet | 16.9 | 21.8 | 7.54 | 0.86 |
| MLP | 16.61 | 21.23 | 7.32 | 0.87 |
| RNN (Vanilla) | 16.15 | 6.94 | 0.87 |
| LSTM | 17.15 | 22.5 | 7.26 | 0.85 | 
| GRU | 17 | 22.12 | 7.32 | 0.86 |





### 2. Development Environment
```
- Windows 10 Home
- Python 3.7.3
- torch 1.8.1
- NVIDIA GFORCE RTX 2060

