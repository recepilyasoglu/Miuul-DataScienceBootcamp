import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("Feature-Engineering/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("Feature-Engineering/titanic.csv")
    return data

df = load()
df.head()

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

# notnull dan sonra True-False dönüyor onu int yaparak 1 veya 0 olmasını sağladık
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# kabini dolu olanların hayatta kalma oranı, NAN olan kabinlere göre daha fazla
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


from statsmodels.stats.proportion import proportions_ztest

# kabin numarası olan ve hayatta kalan kaç kişi
# kabin numarası olmayan ve hayatta kalan kaç kişi
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
# H0 Reddedildi, ikisi arasında fark var
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 0 dan büyük ise yalnız değil,
# kişinin tek olması durumuna göre hayatta kalma çabasını gelişrtirmiş olabilir? bilemiyoruz
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

# H0 Reddedildi, ikisi arasında fark var
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))





