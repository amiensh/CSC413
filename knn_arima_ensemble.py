import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import glob

# The kNN code is based on the knnClassfier algorithm for stock market prediction on medium.com by 10mohi6. Citation is given below
# 10mohi6. (2020, December 3). Super Easy Python stock price forecasting(using K-nearest neighbor) machine learning. Medium. Retrieved April 16, 2023, from https://10mohi6.medium.com/super-easy-python-stock-price-forecasting-using-k-nearest-neighbor-machine-learning-ab6f037f0077 

# The ARIMA code is based on an article written by Cory Maklin in Medium and an article written by Jason Brownlee on machinelearningmastery.com. Cited below
# Maklin, C. (2019, May 25). Arima model python example - time series forecasting. Medium. Retrieved April 17, 2023, from https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7 
# Brownlee, J. (2017, March 24). How to make out-of-sample forecasts with Arima in python. MachineLearningMastery.com. Retrieved April 19, 2023, from https://machinelearningmastery.com/make-sample-forecasts-arima-python/ 

def preprocess(filename: str):
   df = pd.read_csv(filename)
   df = df.dropna()
   targets = df["Close"]
   df.set_index("Date")
   
   return  df, targets

def micron_preprocess(filename: str):
   df = pd.read_csv(filename)
   df = df.dropna()
   
   df["Next"] = np.zeros(len(df))
   for i in range(len(df)-1):
      df["Next"][i] = df["Close"][i+1]
   targets = df["Next"]
   df = df[["Open", "Close", "High", "Low"]]
   
   
   return  df, targets, "Close"

def exchange_preprocess(filename: str):
   df = pd.read_csv(filename)
   df = df.dropna()
   df["Next"] = np.zeros(len(df))
   for i in range(len(df)-1):
      df["Next"][i] = df["OT"][i+1]
   targets = df["Next"]
   df = df[["0","1", "3","OT"]]

   return df, targets, "OT"

def arima_predictions(X_train,column, n_steps):
   model = ARIMA(X_train[column],  order = (2,1,1))
   model_fit = model.fit(method_kwargs={"warn_convergence": False})
   return model_fit.forecast(steps = n_steps)

def knn_predictions(X_train, y_train, X_test):
   model = KNeighborsRegressor(n_neighbors=3)
   model.fit(X_train, y_train)
   return model.predict(X_test)


if __name__ == '__main__':
   path = "./exchange_rate.csv"
   files = glob.glob(path)

   arima_mse = 0
   arima_mae = 0

   knn_mse = 0
   knn_mae = 0

   ensemble_mse = 0
   ensemble_mae = 0
   # Uncomment one of the 2 depending on the dataset

   data, targets, column = exchange_preprocess(files[0])
   #data, targets, column = micron_preprocess(files[0])

   test_data = data.tail(1)
   test_target = data.tail(1)[column]
   X_train = data.drop(data.tail(1).index)
   y_train = targets.drop(targets.tail(1).index)

   print("test_target: ", test_target)
   
   
   pred_arima = arima_predictions(X_train, column, 1)
   print("ARIMA pred: ", pred_arima)
   pred_knn = knn_predictions(X_train, y_train,test_data)
   print("kNN pred: ", pred_knn)
   pred = (pred_arima + pred_knn)/2

   arima_mse = mean_squared_error(pred_arima,test_target)
   arima_mae = mean_absolute_error(pred_arima,test_target)

   knn_mse = mean_squared_error(pred_knn,test_target)
   knn_mae = mean_absolute_error(pred_knn,test_target)

   ensemble_mse = mean_squared_error(pred,test_target)
   ensemble_mae = mean_absolute_error(pred,test_target)

   print("ARIMA mse loss: ", arima_mse)
   print("ARIMA mae loss: ", arima_mae)

   print("kNN mse loss: ", knn_mse)
   print("kNN mae loss: ", knn_mae)

   print("Ensemble mse loss: ", ensemble_mse)
   print("Ensemble mae loss: ", ensemble_mae)

