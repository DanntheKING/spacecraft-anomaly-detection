import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_channel_data(data_dir, channel_id):
    labels = pd.read_csv(f"{data_dir}/labeled_anomalies.csv")

    channel_info = labels[labels["chan_id"] == channel_id].iloc[0]

    train_channel = np.load(f"{data_dir}/train/{channel_id}.npy")
    test_channel = np.load(f"{data_dir}/test/{channel_id}.npy")

    y_true = np.zeros(len(test_channel))

    ranges = ast.literal_eval(channel_info["anomaly_sequences"])

    for start, end in ranges:
        y_true[start:end + 1] = 1

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_channel)
    X_test = scaler.transform(test_channel)

    return X_train, X_test, y_true, scaler
