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


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

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


####################################
# Label Encoding & Binary Encoding
####################################

df["Sex"].head()
le = LabelEncoder()

# ilk 5 satırını 0 ve 1'lere dönüştür.
# Alfabetik sıralamaya göre ilk gördüğüne 0 verir, mesela bu örnek de female 0 oldu
le.fit_transform(df["Sex"])[0:5]

# hangisine 0 hangisine 1 verdiğimizi unuttuğumuz bir senaryoda
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# elimde yüzlerce değişken olduğu senaryo da
# değişkenin sütunlarında gez,
# tipine bak eğer int ve float olmayan(int ise mesela o zaten binary edilmiş)
# ve eşsiz sınıf sayısı 2 olanları seç
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()
# df["FLAG_EMP_PHONE"].value_counts()
# df["EMERGENCYSTATE_MODE"].value_counts()

for col in binary_cols:
    label_encoder(df, col)

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())


