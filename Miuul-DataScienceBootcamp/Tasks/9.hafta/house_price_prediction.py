###########################################
############# RECEP İLYASOĞLU #############
###########################################

# İş Problemi

# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
# gerçekleştirilmek istenmektedir.


# Veri Seti Hikayesi

# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması
# da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait
# olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
# değerleri sizin tahmin etmeniz beklenmektedir.
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation


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

df = pd.concat([train, test])

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


def grab_col_names(dataframe, cat_th=10, car_th=20):
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
                   (dataframe[col].dtypes != "O") and (dataframe[col].dtypes != "category")]
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

df[cat_but_car]

df[num_cols].dtypes

df["Neighborhood"]

df[cat_cols].dtypes

cat_cols.append(cat_but_car[0])
df[cat_cols] = df[cat_cols].astype("category")

# Id ve SalePrice değişkenlerini num_cols dan çıkardım
num_cols = num_cols[1:]
num_cols = num_cols[:-1]

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

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    col = pd.to_numeric(dataframe[col_name], errors='coerce')  # numeric tipine dönüşüm için
    quartile1 = col.quantile(q1)
    quartile3 = col.quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# hata aldığım için check_outlier fonksşiyonunu yapılandırdım
# dataframe[col_name] ifadesi kategorik bir sütunsa ve categorical tipindeyse hataya neden oluyordu.
# onun için ilk olarak if = category dedim
def check_outlier(dataframe, col_name):
    if dataframe[col_name].dtype != 'category':
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        col = pd.to_numeric(dataframe[col_name], errors='coerce')  # numeric tipine dönüşüm için
        if col[(col > up_limit) | (col < low_limit)].any(axis=None):
            return True
        else:
            return False

for col in num_cols:
    print(col, check_outlier(df, col))


# sns.set_style("whitegrid")
# sns.boxplot(data=df[num_cols], orient="h", palette="Set2")

# fig, ax = plt.subplots(figsize=(10,5))
# ax.boxplot(train['SalePrice'])
# ax.set_title('Boxplot of SalePrice')
# plt.show()


# Adım 7: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.isnull().sum().sum()


## Görev 2: Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# Aykırı Değerler
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    col = pd.to_numeric(dataframe[variable], errors='coerce')  # numeric tipine dönüşüm için
    dataframe.loc[(col < low_limit), variable] = low_limit
    dataframe.loc[(col > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

# df.groupby(cat_cols)["TotalCharges"].mean()


# Eksik Değerler
df[num_cols].dtypes

for col in num_cols:
    if df[col].dtype == "int64":
        df[col] = df[col].fillna(df.groupby("OverallQual")[col].transform("mean").round())
    else:
        df[col] = df[col].fillna(df.groupby("OverallQual")[col].transform("mean"))

df["FireplaceQu"]
df["MasVnrType"]
# na_values = ["FireplaceQu", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", \
#              "BsmtFinType1", "GarageCond", "GarageYrBlt", "GarageFinish", \
#              "GarageQual", "GarageType"]

# df[na_values].isnull().sum()

df[num_cols].isnull().sum()

# burada ki değişkenler mesela MasVnrType = Duvar kaplama tipi gibi değişkenleri en çok kullanılan değerler ile değiştirdim

df[cat_cols].isnull().sum()
def fill_na_values_with_mode(dataframe, columns):
    for col in columns:
        # print(dataframe[col])
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode().iloc[0])

fill_na_values_with_mode(df, cat_cols)

missing_values_table(df)

# df.dropna(inplace=True)

# Adım 2: Rare Encoder uygulayınız.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)


# 0.01 rare oranının altında kalan kategorik değişken sınıflarını bir araya getirecek
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

new_df

# Adım 3: Yeni değişkenler oluşturunuz.

def set_type(dataframe, col, metric):
    dataframe[col] = dataframe[col].astype(metric)

# yapı yaşı
new_df["Age_Building"] = new_df["YearRemodAdd"] - new_df["YearBuilt"]

# ev kalitesi
new_df['Total_Home_Quality'] = new_df['OverallQual'] + new_df['OverallCond'].astype(float)

# toplam alan
new_df['TotalArea'] = new_df['1stFlrSF'] + new_df['2ndFlrSF'] + new_df['TotalBsmtSF'] + + new_df['GarageArea']

# yaşam alanı oranı
new_df['LivingAreaRatio'] = new_df['GrLivArea'] / new_df['TotalArea']

# toplam banyo sayısı
set_type(new_df, ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"], int)
new_df['TotalBathrooms'] = new_df['BsmtFullBath'] + new_df['BsmtHalfBath'] + new_df['FullBath'] + new_df['HalfBath']

# garaj kapasitesi
set_type(new_df, "GarageCars", float)
new_df['GarageCapacity'] = new_df['GarageCars'] + new_df['GarageArea']


# Adım 4: Encoding işlemlerini gerçekleştiriniz.

# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in new_df.columns if new_df[col].dtype not in ["int64", "float64"] and new_df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(new_df, col)

new_df[binary_cols]

# One Hot Encoding
ohe_cols = [col for col in new_df.columns if 10 >= new_df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


new_df = one_hot_encoder(new_df, ohe_cols)

