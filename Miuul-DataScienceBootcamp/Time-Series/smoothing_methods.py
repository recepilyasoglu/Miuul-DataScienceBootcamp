####################################
# Smoothing Methods (Holt-Winters) #
####################################

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.filterwarnings('ignore')


##################
# Veri Seti
##################

# Amaç: Bir sonraki zaman periyodunda hava kirliliğinin ne olabileceğini tahmin etmek

data = sm.datasets.co2.load_pandas()

y = data.data  # bağımlı değişkenimiz

# veriler haftalık veriler hava kirliliğini aylık ölçmemiz lazım
# ilk iş aylara göre groupby

y = y["co2"].resample("MS").mean()  # aylık frekansa göre ortalama al

# eksik değerleri doldurma
y.isnull().sum()

y = y.fillna(y.bfill())  # bir sonraki değer ile doldurma

y.plot(figsize=(15, 6))
plt.show()













