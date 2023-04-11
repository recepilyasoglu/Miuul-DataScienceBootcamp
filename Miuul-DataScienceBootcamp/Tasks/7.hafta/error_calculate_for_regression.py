###########################################
############# RECEP İLYASOĞLU #############
###########################################

############ Error Evaluation for Regression Models ############

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', None)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


# Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.

df = pd.DataFrame(data={"Deneyim Yılı (x)": (5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1),
                       "Maaş (y)": (600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380)})

df.head()

# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
## Bias = 275, Weight= 90 (y’ = b+wx)

X = df[["Deneyim Yılı (x)"]]
y = df[["Maaş (y)"]]

reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
reg_model.intercept_[0]

# maaş'ın katsayısı (w1)
reg_model.coef_[0][0]


# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.

y_pred = reg_model.predict(X)

df["Maaş Tahmini (y')"] = y_pred


# 3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız
#mse
mean_squared_error(y, y_pred)

#rmse
np.sqrt(mean_squared_error(y, y_pred))

#mae
mean_absolute_error(y, y_pred)

df["Hata (y-y'))"] = (df["Maaş (y)"] - df["Maaş Tahmini (y')"])
df["Hata Kareleri"] = (df["Hata (y-y'))"]**2)
df["Mutlak Hata (|y-y'|)"] = abs(df["Hata Kareleri"])

# del df["Maaş Tahmini"]
