###########################################
############# RECEP İLYASOĞLU #############
###########################################

############ Telco Churn Prediction ############

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

df = pd.read_csv("Tasks/9.hafta/Telco-Customer-Churn.csv")
df.head()

df.shape
df.describe().T
df.isnull().sum() / df.shape[0] * 100
df.info()
df.dtypes


## Görev 1 : Keşifçi Veri Analizi

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

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

num_cols
cat_cols
cat_but_car


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

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["SeniorCitizen"]

df["customerID"]

cat_cols
df[cat_cols]


# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

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

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

cat_cols = cat_cols[:-1]
num_cols = num_cols[1:]

df.groupby("Churn")[cat_cols].count()

# Adım 5: Aykırı gözlem var mı inceleyiniz.

df[num_cols].describe().T


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, num_cols)


def check_outlier(dataframe,
                  col_name):  # q1 ve q3 'ü de biçimlendirmek istersek check_outlier'a argüman olarak girmemiz gerekir
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, num_cols))


# Adım 6: Eksik gözlem var mı inceleyiniz.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

df.isnull().sum()

## Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df.groupby(cat_cols)["TotalCharges"].mean()

for col in num_cols:
    # print(df[col])
    df[col] = df[col].fillna(df.groupby(cat_cols)[col].transform("mean"))

df.dropna(inplace=True)

df.corr().sort_values("Churn", ascending=False) \
    .drop("Churn", axis=0)

# Adım 2: Yeni değişkenler oluşturunuz.

df.columns

# df["Num_Gender"] = [0 if col == "Male" else 1 for col in df.gender]
# df[["gender", "Num_Gender"]].head(20)

df["New_Tenure"] = pd.cut(df["tenure"], bins=[0, 10, 15, 72], labels=["New", "Star", "Loyal"])
df[["New_Tenure", "tenure"]].head(20)

df[["New_Tenure", "PaymentMethod"]].head(20)
df.groupby(["New_Tenure", "PaymentMethod"]).agg({"PaymentMethod": "count"})

df["ContractLength"] = np.where(df["Contract"] == "Month-to-month", "Short", "Long")
df[["New_Tenure", "tenure"]].head(20)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
new_variables = df[["New_Tenure", "ContractLength"]]


def count_of_values(dataframe):
    for col in dataframe:
        print(dataframe[col].value_counts())


count_of_values(new_variables)

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()
df
# One Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ohe_cols)
df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
for col in num_cols:
    print(col, check_outlier(df, col))

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

## Görev 3 : Modelleme

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

random_user = X.sample(1, random_state=45)  # rastgele bir kullanıcı oluşturuyoruz

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Logistic Regression
log_model = LogisticRegression().fit(X, y)

log_cv_results = cross_validate(log_model,
                                X, y,  # bağımlı ve bağımsız değişkenler
                                cv=5,  # dördüyle model kur, biriyle test et
                                scoring=["accuracy", "f1", "roc_auc"])  # istediğimiz metrikler

log_test = log_cv_results['test_accuracy'].mean()
# 0.8059382470763069
log_f1 = log_cv_results['test_f1'].mean()
# 0.5911821068005934
log_auc = log_cv_results['test_roc_auc'].mean()
# 0.8463000059038415
log_model.predict(random_user)

# RandomForestClassifier
rf_model = RandomForestClassifier().fit(X, y)

rf_cv_results = cross_validate(rf_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

rf_test = rf_cv_results['test_accuracy'].mean()
# 0.7910226666989727
rf_f1 = rf_cv_results['test_f1'].mean()
# 0.5533446100583161
rf_auc = rf_cv_results['test_roc_auc'].mean()
# 0.8244377769644089
rf_model.predict(random_user)

# GBM
gbm_model = GradientBoostingClassifier().fit(X, y)

gbm_cv_results = cross_validate(gbm_model, X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

gbm_test = gbm_cv_results['test_accuracy'].mean()
# 0.8018179193319119
gbm_f1 = gbm_cv_results['test_f1'].mean()
# 0.582522324396216
gbm_auc = gbm_cv_results['test_roc_auc'].mean()
# 0.8455268626584862

# LightGBM
lgbm_model = LGBMClassifier().fit(X, y)

lgbm_cv_results = cross_validate(lgbm_model,
                                 X, y,
                                 cv=5,
                                 scoring=["accuracy", "f1", "roc_auc"])

lgbm_test = lgbm_cv_results['test_accuracy'].mean()
# 0.7938640805711701
lgbm_f1 = lgbm_cv_results['test_f1'].mean()
# 0.574774214849072
lgbm_auc = lgbm_cv_results['test_roc_auc'].mean()
# 0.833585885238031
lgbm_model.predict(random_user)

# XGBoost
xgboost_model = XGBClassifier(use_label_encoder=False)

xg_cv_results = cross_validate(xgboost_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

xg_test = xg_cv_results['test_accuracy'].mean()
# 0.7815045107255928
xg_f1 = xg_cv_results['test_f1'].mean()
# 0.5535958862511531
xg_auc = xg_cv_results['test_roc_auc'].mean()
# 0.8201984479635728


# K-NN
knn_model = KNeighborsClassifier().fit(X, y)

# cross validation
knn_cv_results = cross_validate(knn_model,
                                X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

knn_test = knn_cv_results['test_accuracy'].mean()
# 0.7736917078568198
knn_f1 = knn_cv_results['test_f1'].mean()
# 0.5573913274733598
knn_auc = knn_cv_results['test_roc_auc'].mean()
# 0.7887149853886167
knn_model.predict(random_user)

# Decision Tree
dt_model = DecisionTreeClassifier().fit(X, y)

dt_cv_results = cross_validate(dt_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])
dt_cv_results
dt_test = dt_cv_results['test_accuracy'].mean()
# 0.7340550696194353
dt_f1 = dt_cv_results['test_f1'].mean()
# 0.50013672303396
dt_auc = dt_cv_results['test_roc_auc'].mean()
# 0.6606732044384354

dt_model.predict(random_user)

best_model_results = pd.DataFrame(
    {"Model": ["Logistic Regression", "Random Forest", "GBM", "LightGBM", "XGBoost", "KNN", "Decision Tree"],
     "Accuracy": [log_test, rf_test, gbm_test, lgbm_test, xg_test, knn_test, dt_test],
     "AUC": [log_auc, rf_auc, gbm_auc, lgbm_auc, xg_auc, knn_auc, dt_auc]},
    index=range(1, 8))

best_model_results.sort_values("Accuracy", ascending=False)

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin
# ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.








