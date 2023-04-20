from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import glob
import matplotlib.pyplot as plt
import time

def exchange_preprocess(filename: str):
   df = pd.read_csv(filename)
   df = df.dropna()
   targets = df["OT"]

   for i in range(len(df)):
       df["date"][i] = time.mktime(time.strptime(df["date"][i],"%Y/%m/%d %H:%M"))
   
   df = df["date"]
   return  df, targets

def micron_preprocess(filename: str):
   df = pd.read_csv(filename)
   df = df.dropna()
   targets = df["Close"]

   for i in range(len(df)):
       df["Date"][i] = time.mktime(time.strptime(df["Date"][i],"%Y-%m-%d"))
   df = df["Date"]

   
   return  df, targets

if __name__ == "__main__":
    path = "./exchange_rate.csv"
    files = glob.glob(path)
    knn_mse_losses = []
    arima_mse_losses = []
    ensemble_mse_losses = []
   
    knn_mae_losses = []
    arima_mae_losses = []
    ensemble_mae_losses = []

    data, targets = micron_preprocess(files[0])

    test_data = data.tail(96)
    test_target = targets.tail(96)
    X_train = data.drop(data.tail(96).index)
    y_train = targets.drop(targets.tail(96).index)

    print("X_train.shape: ", X_train.shape)
    model = LinearRegression().fit(np.reshape(np.array(X_train),(-1,1)), y_train)
    pred = model.predict(np.reshape(np.array(test_data),(-1,1)))

    print("MSE loss: ", mean_squared_error(pred, test_target))
    print("MAE loss: ", mean_absolute_error(pred, test_target))

    plt.scatter(test_target,pred)
    plt.plot(test_target, test_target, color = 'orange')
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Value")
    plt.title("Predicted vs Actual")
    plt.show()
