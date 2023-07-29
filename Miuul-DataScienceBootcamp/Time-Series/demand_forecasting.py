#####################################################
# Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge
# https://www.kaggle.com/c/demand-forecasting-kernels-only

# Amaç mağaza ürün kırılımında 3 ay sonrasını tahmin etmek

import matplotlib
matplotlib.use("Qt5Agg")
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv("Time-Series/datasets/demand_forecasting/train.csv", parse_dates=["date"])
test = pd.read_csv("Time-Series/datasets/demand_forecasting/test.csv", parse_dates=["date"])

sample_sub = pd.read_csv("Time-Series/datasets/demand_forecasting/sample_submission.csv")

df = pd.concat([train, test], sort=False)  # eda ve feature işlemlerini tek bir dataframe üzerinde yapmak daha iyi olabilir,
# eksilerinden biri data lekage, veri sızıntısı olabilir, tabular veri seti olsaydı daha dikkatli olurduk


#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()

check_df(df)

df[["store"]].nunique()

df[["item"]].nunique()

df.groupby(["store"])["item"].nunique()

df.groupby(["store", "item"]).agg({"sales": ["sum"]})

df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.head()


#####################################################
# FEATURE ENGINEERING
#####################################################

# hangi alan üzerinde çalışıyorsak bu alanın takvim bilgisine sahip olmak zorundayız
# ilgili talep-tahmin modeline bu takvimi yerdirmek/yansıtmak zorundayız
# ona göre bir feature üretip eklemek zorundayız bir ayırt ediciliği olması için
# bazı mevsimsellik(tekrarlanan) feature'lar yeterli patern'liğe(örüntüye) sahip olmayabilir
# Örn: sevgililer günü veri setin 3 yıllık ise 3 defa sevgililer gününün çok katkısı olmaz

df.head()

def create_date_features(dataframe):
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.weekofyear
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek
    dataframe['year'] = dataframe.date.dt.year
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    return dataframe

df = create_date_features(df)
df.head()

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})




























