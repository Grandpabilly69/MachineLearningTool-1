import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
df

y = df["logS"]
y

x = df.drop("logS", axis=1)
x

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train

x_test

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

y_lr_train_pred
y_lr_test_pred

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(("Linear regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2)).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]
lr_results

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(("Random forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2)).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]
rf_results

df_models = pd.concat((lr_results, rf_results), axis=0)
df_models

df_models.reset_index(drop=True)


plt.figure(figsize=(5, 5))
plt.scatter(x = y_train, y = y_lr_train_pred, alpha = 0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), "#F8766D")
plt.ylabel("Predict logS")
plt.xlabel("Actual logS")