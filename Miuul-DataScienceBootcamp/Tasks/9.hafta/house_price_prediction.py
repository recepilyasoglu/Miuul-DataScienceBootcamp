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

df = pd.concat([test.assign(ind="test"), train.assign(ind="train")])

# test, train = df[df["ind"].eq("test")], df[df["ind"].eq("train")]

df.head()
df.shape
df.describe().T
df.dtypes
df.isnull().sum()

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

# def get_col_types(dataframe, cat_th=10, car_th=10):
#     var_types = dataframe.dtypes
#
#     # Kategorik, numerik ve numerik fakat kategorik değişkenleri ayır
#     categorical_cols = list(var_types[var_types == 'object'].index)
#     numeric_cols = list(var_types[var_types == 'float64'].index) + list(var_types[var_types == 'int64'].index)
#
#     num_but_cat = []
#     cat_but_car = []
#
#     for col in categorical_cols:
#         unique_vals = len(str(df[col].nunique()))
#         if unique_vals > car_th:  # Burda eşsiz değer sayısı 10'dan fazla ise kardinal olarak kabul ettim
#             cat_but_car.append(col)
#
#     for col in numeric_cols:
#         unique_vals = len(str(df[col].nunique()))
#         if unique_vals <= cat_th:  # Burda da eşsiz değer sayısı 10 veya daha az ise numerik görünümlü kategorik olarak kabul ettim
#             categorical_cols.append(col)
#             num_but_cat.append(col)
#
#     # Sonuçlar
#     print(f"Observations: {df.shape[0]}")
#     print(f"Variables: {df.shape[1]}")
#     print(f'cat_cols: {len(categorical_cols)}')
#     print(f'num_cols: {len(numeric_cols)}')
#     print(f'cat_but_car: {len(cat_but_car)}')
#     print(f'num_but_cat: {len(num_but_cat)}')
#     return categorical_cols, numeric_cols, cat_but_car, num_but_cat
#
# cat_cols, num_cols, cat_but_car, num_but_cols = get_col_types(df)


def grab_col_names(dataframe, cat_th=2, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


df[cat_cols].dtypes
df[num_cols].dtypes
df[cat_but_car].dtypes

def get_stats(dataframe, col):
    return print("############### İlk 5 Satır ############### \n", dataframe[col].head(), "\n", \
                 "############### Sahip olduğu Değer Sayısı ############### \n", dataframe[col].value_counts(), "\n", \
                 "############### Toplam Gözlem Sayısı ############### \n", dataframe[col].shape, "\n", \
                 "############### Değişken Tipleri ############### \n", dataframe[col].dtypes, "\n", \
                 "############### Toplam Null Değer Sayısı ############### \n", dataframe[col].isnull().sum(), "\n", \
                 "############### Betimsel İstatistik ############### \n", dataframe[col].describe().T
                 )


get_stats(df, cat_cols)
get_stats(df, num_cols)

# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["Neighborhood"] = df["Neighborhood"].astype("category")

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

df.groupby("SalePrice")[cat_cols].count()

# Adım 6: Aykırı gözlem var mı inceleyiniz.

num_cols = num_cols[1:]
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, num_cols)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, num_cols))


sns.set_style("whitegrid")
sns.boxplot(data=df[num_cols], orient="h", palette="Set2")

fig, ax = plt.subplots(figsize=(10,5))
ax.boxplot(train['SalePrice'])
ax.set_title('Boxplot of SalePrice')
plt.show()


# Adım 7: Eksik gözlem var mı inceleyiniz.



