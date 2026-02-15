# Primitive BTC/USD price prediction
<img width="1918" height="1033" alt="Figure_1" src="https://github.com/user-attachments/assets/31c89ccf-9f5c-4bef-886e-e2b48dacb5d0" />

Outputs a matplot graph showing the historical and predicted prize in USD in red and green respectively.

## Dataset
I used the following dataset by [Mark Zielinski](https://github.com/mczielinski): https://github.com/mczielinski/kaggle-bitcoin

I renamed the header and only kept one date:
```csv
date,open,high,low,close,volume,market_cap
```

## Goal

Predict the value of BTC in USD and laugh about its precision (dataset is about a year old)

## Training

I tried a lot of methods for training the model, I landed on using Tensorflow along with some scikit functions (originally I used only scikit, but the produced results were... really bad, even though the tests had wonderful accuraccy). I planned on using PolynomialFeatures to help predict the price because of its volatility, but I don't have the hardware to train using that.

Specifically, I'm using LSTM models, which are a pain to use, because I constantly need to reshape the input and outputs to and from these models, but they don't require a more complicated setup.

## Is it successful?

Kind of, the output looks plausible at first glance, but it is not applicable at all. It does, however, perform as I expected, and so it does reach the goal it was designed for.
