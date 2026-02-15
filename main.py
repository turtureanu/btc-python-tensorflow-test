import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Tensorflow requires numpy arrays everywhere :[

SEQ_LEN = 365  # sequence (a year) to predict the next day
FUTURE_DAYS = 365  # days to predict
EPOCHS = 100  # Epochs to run for

np.random.seed(42)
tf.random.set_seed(42)

CSV_PATH = "./test.csv"

# Sort data by date
df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date")
df.set_index("date", inplace=True)

features = df[["open", "high", "low", "close", "volume"]].values.copy()

train_price, test_price = train_test_split(
    features,
    test_size=0.2,
    shuffle=False,  # cannot shuffle future and past, 80/20 is pretty standard
)

# scale down the price <0; 1>
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

train_scaled = feature_scaler.fit_transform(train_price)
test_scaled = feature_scaler.transform(test_price)

# scale ONLY the close column separately
train_close_scaled = target_scaler.fit_transform(train_price[:, 3].reshape(-1, 1))
test_close_scaled = target_scaler.transform(test_price[:, 3].reshape(-1, 1))

# TRAIN

X_train = []
y_train = []

# get sequences and their target value (price)
for i in range(SEQ_LEN, len(train_scaled) - FUTURE_DAYS):
    seq = train_scaled[i - SEQ_LEN : i]
    X_train.append(seq)

    # next FUTURE_DAYS days' "close" value for the sequence
    y_train.append(train_close_scaled[i : i + FUTURE_DAYS, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# TEST

X_test = []
y_test = []

for i in range(SEQ_LEN, len(test_scaled) - FUTURE_DAYS):
    seq = test_scaled[i - SEQ_LEN : i]
    X_test.append(seq)

    y_test.append(test_close_scaled[i : i + FUTURE_DAYS, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

# https://www.tensorflow.org/guide/keras/sequential_model
# https://www.statology.org/how-to-build-lstm-models-for-time-series-prediction-in-python/
model = Sequential(
    [
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(SEQ_LEN, features.shape[1]),  # return sequences for next LSTM
        ),  # predict price
        LSTM(units=64),
        Dense(FUTURE_DAYS),  # the values for the next FUTURE_DAYS
    ]
)

model.compile(
    optimizer="adam", loss="mean_squared_error"
)  # got this from the previous guide

model.summary()  # prints the model architecture

# from previous guide
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )  # scikit did this automatically, https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    ],
)

# FUTURE PREDICTIONS

scaled_features = feature_scaler.transform(features)
last_seq = scaled_features[
    -SEQ_LEN:
].reshape(  # get last FUTURE_DAYS and convert/reshape to LSTM format
    1, SEQ_LEN, features.shape[1]
)  # we need to reshape because of LSTM (batch_size, seq_size, features - price)

future_scaled = model.predict(last_seq, verbose=0)[0].reshape(
    -1, 1
)  # predict future prices

future_forecast = target_scaler.inverse_transform(future_scaled)
# add an offset (because it doesn't work normally?)
future_forecast += df["close"].iloc[-1] - future_forecast[0]

# date range (end of dataset z FUTURE_DAYS)
future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS
)


# PLOT

plt.figure(figsize=(14, 6))

# historical price in red
plt.plot(df.index, df["close"], label="Close", linewidth=2, color="red")

# future forecast in green
plt.plot(
    future_dates,
    future_forecast,
    label=f"{FUTURE_DAYS}-Day LSTM Forecast",
    linewidth=2,
    color="green",
)

plt.title("Bitcoin/USD")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.grid(True)
plt.tight_layout()
plt.show()
