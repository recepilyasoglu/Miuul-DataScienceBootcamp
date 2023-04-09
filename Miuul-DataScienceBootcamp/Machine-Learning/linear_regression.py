######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
df = pd.read_csv("Machine-Learning/Datasets/advertising.csv")
df.head()
df.shape

X = df[["TV"]]  # bağımsız değişken
y = df[["sales"]]  # bağımlı değişken


##########
# Model
##########
reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]

##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# 500 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

df.describe().T

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51
y.mean()  # alınan MSE değerinin ne olduğunu anlayabilmek adına bağımlı değikenin ortalamasına ve
y.std()  # standart sapmasına baktık, sonuç olarak büyük bir değer(10.51) geldi bu örneğe göre

# RMSE
np.sqrt(mean_squared_error(y, y_pred))  # yukarıdan gelen ifadenin karekökü
# 3.24

# MAE
mean_absolute_error(y, y_pred)  # bu daha düşük geldi daha mı iyi ?
                                # hayır bunu anca model de diyelim ki değişiklik yaptık, bu değişiklikten öncesi ve sonrası
# 2.54                          # değerlendirilir yani değişiklik öncesi MAE ve değişiklik sonrası MAE olarak

# R-KARE    # veri setindeki bağımsız değişkenlerin, bağımlı değişkeni açıklama yüzdesidir.
reg_model.score(X, y)  # bağımsız değişkenlerin, bağımlı değişkenin %61'ini açıklar.


######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("Machine-Learning/Datasets/advertising.csv")

X = df.drop('sales', axis=1)  # bağımsız değişkenler için bağımlı değişken dışındakileri aldık

y = df[["sales"]]  # bağımlı değişken

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

# denklem (mülakat sorusu !!!)
2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN R-KARE
reg_model.score(X_train, y_train)

# Test RMSE     # test hatası, train hatasına göre daha yüksek çıkar normal de, bizde daha düşük çıktı bu güzel bir durum
y_pred = y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# TEST R-KARE
reg_model.score(X_test, y_test)


# veri boyutumuz az lduğu Cross Validation daha çok güvenebiliriz
# boyutumuz çok olsaydı farketmez diyebilirdik.
# 10 Katlı CV RMSE (10 katlı Cross Validation(Çapraz Doğrulama) Skoru)
np.mean(np.sqrt(-cross_val_score(reg_model,  # eksi değerler geldiği için - hatalar olmaz diyerek, - ile çarptık
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


###############################################################
# Simple Linear Regression with Gradient Descent from Scratch #
###############################################################

# Cost function - MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    # ilk hatanın raporlandığı bölüm
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("Machine-Learning/Datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

