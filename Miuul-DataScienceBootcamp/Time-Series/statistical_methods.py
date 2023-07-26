##################################################
# Statistical Methods
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':]


##############################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average) #
##############################################################

# order= da p, d, q değerlerini giriyoruz
arima_model = ARIMA(train, order=(1, 1, 1)).fit()


# istatistiki durumu
arima_model.summary()

y_pred = arima_model.forecast(steps=48)  # 48 birim sonrasına git
y_pred = pd.Series(y_pred, index=test.index)


mae = mean_absolute_error(test, y_pred)

def plot_co2(train, test, y_pred, title):
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "ARIMA")


##############################################################
# Hyperparameter Optimization (Model Derecelerini Belirleme) #
##############################################################

##############################################################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme #
##############################################################

p = d = q = range(0, 4)

# olası kombinasyonları çıkarttık, bunlar da gezmemiz gerekiyor ve en iyisini seçmemiz gerkiyor
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model_result = ARIMA(train, order=order).fit()
            aic = arima_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except Exception as e:
            print('Error for ARIMA%s: %s' % (order, e))
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)


############################
# Final Model
############################

arima_model = ARIMA(train, order=best_params_aic).fit()
y_pred = arima_model.forecast(steps=48)

y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")




