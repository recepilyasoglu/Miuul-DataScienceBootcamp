#####################################################
################## RECEP İLYASOĞLU ##################
#####################################################

# Customer Segmentation with Unsupervised Learning


## Business Problem:
# FLO segments its customers and according to these segments
# Wants to define marketing strategies. for this as a result, the behavior of customers will be defined and this
# Groups will be formed according to clusters in behavior.
# (FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik
# olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.)


## Dataset Story:
# The dataset includes the last shopping from Flo to OmniChannel (both online and offline shoppers) in 2020 - 2021.
# It consists of information obtained from the past shopping behavior of customers who
# (Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır)


import joblib
import warnings
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

# Task 1: Preparing the Data

# Adım 1: flo_data_20K.csv verisini okutunuz.

data = pd.read_csv("Tasks/10.hafta/flo_data_20k.csv")
df = data.copy()
df.head()
df.isnull().sum()
df.describe().T

df["order_channel"].value_counts()

# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# Aykırı değer kontrolü
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

# sütunlar da kontrol ettim, olanları liste içerisine alıp baskılayabilmek adına
outlier_thresholds(df, df.columns)

values = ["customer_value_total_ever_offline", "customer_value_total_ever_online",
          "customer_value_total_ever_offline", "customer_value_total_ever_online"]
def check_outlier(dataframe, col_name):
    if dataframe[col_name].dtype != 'category':
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        col = pd.to_numeric(dataframe[col_name], errors='coerce')
        if col[(col > up_limit) | (col < low_limit)].any(axis=None):
            return True
        else:
            return False

for col in values:
    print(col, check_outlier(df, col))
    # sağlamasında da görmüş oldum yukarıdaki liste eçerisinde yer alan değerler de aykırılık var

def replace_with_threshold(dataframe, variable):
    low_limit, upl_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > upl_limit), variable] = upl_limit

for col in values:
    replace_with_threshold(df, col)








