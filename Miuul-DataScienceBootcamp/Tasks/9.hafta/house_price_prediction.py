###########################################
############# RECEP İLYASOĞLU #############
###########################################

############ House Price Prediction Model ############

import warnings
import matplotlib
matplotlib.use("Qt5Agg")
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

## Görev 1: Keşifçi Veri Analizi

# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.

test = pd.read_csv("Tasks/9.hafta/test.csv")
test.head()
train = pd.read_csv("Tasks/9.hafta/train.csv")
train.head()

df = pd.concat([train, test], axis=1)
df.head()
df.shape
df.describe().T
df.dtypes
df.isnull().sum()

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def get_col_types(dataframe, cat_th=10, car_th=10):
    var_types = dataframe.dtypes

    # Kategorik, numerik ve numerik fakat kategorik değişkenleri ayır
    categorical_cols = list(var_types[var_types == 'object'].index)
    numeric_cols = list(var_types[var_types == 'float64'].index) + list(var_types[var_types == 'int64'].index)

    num_but_cat = []
    cat_but_car = []

    for col in categorical_cols:
        unique_vals = len(df[col].nunique())
        if unique_vals > car_th:  # Burda eşsiz değer sayısı 10'dan fazla ise kardinal olarak kabul ettim
            cat_but_car.append(col)

    for col in numeric_cols:
        unique_vals = len(str(df[col].nunique()))
        if unique_vals <= cat_th:  # Burda da eşsiz değer sayısı 10 veya daha az ise numerik görünümlü kategorik olarak kabul ettim
            categorical_cols.append(col)
            num_but_cat.append(col)

    # Sonuçlar
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(categorical_cols)}')
    print(f'num_cols: {len(numeric_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return categorical_cols, numeric_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cols = get_col_types(df)

df[cat_cols].dtypes
df[num_cols].dtypes
df[num_but_cols].dtypes

# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")

# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###############################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, True)

# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

# Adım 6: Aykırı gözlem var mı inceleyiniz.

# Adım 7: Eksik gözlem var mı inceleyiniz.



