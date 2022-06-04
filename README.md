# Short-Term-Electricity-Price-Forecasting-at-the-Polish-Power-Exchange-Day-Ahead-Market


### 1. Overview
This repository contains the experimental source code for short term electricity price forecasting at the Polish Power Exchange Day Ahead Market  including RNN, LSTM, GRU, MLP and Prophet models.

According to the table below, **RNN** outperformed the other models

| Model | MAE↓ | RMSE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Prophet | 16.9 | 21.8 | 7.54 | 0.86 |
| MLP | 16.61 | 21.23 | 7.32 | 0.87 |
| RNN (Vanilla) | 16.15 | 6.94 | 0.87 |
| LSTM | 17.15 | 22.5 | 7.26 | 0.85 | 
| GRU | 17 | 22.12 | 7.32 | 0.86 |


According to the table below, **DNN** outperformed the other models on **multi-step** time series prediction.
| Model | MAE↓ | MSE↓ | RMSE↓ | MPE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DNN** | **31.3555** | 2913.6521 | **49.3946** | **-16.7329** | **29.1459** | **0.1775** |
| CNN | 32.9762 | **2893.2201** | 49.5900 | -21.7513 | 32.3016 | 0.1206 |
| RNN | 32.9153 | 2951.9055 | 50.0931 | -20.7460 | 32.2081 | 0.1223 |
| LSTM | 32.8141 | 2955.5278 | 50.1237 | -20.5471 | 32.0873 | 0.1191 |
| GRU | 33.0092 | 2927.5575 | 49.9503 | -21.2869 | 32.5345 | 0.1177 |
| Attentional LSTM | 32.2182 | 2920.8744 | 49.7972 | -19.1188 | 30.8223 | 0.1347 |



### 3. Development Environment
```
- Windows 10 Home
- Python 3.7.3
- torch 1.8.1
- NVIDIA GFORCE RTX 2060

