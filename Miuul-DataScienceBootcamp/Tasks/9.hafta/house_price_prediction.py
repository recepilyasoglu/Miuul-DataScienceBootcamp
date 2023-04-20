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
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
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
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   (dataframe[col].nunique() < cat_th) and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   (dataframe[col].nunique() > car_th) and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

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

# SalePrice ve Id değişkenlerini num_cols'dan çıkardım
num_cols = num_cols[1:-1]


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
# df[cat_cols] = df[cat_cols].astype("object")

# kategorik değikenin içerisinde olup numerik tipte olan değişkenleri num_cols'a attım
len(cat_cols)

[num_cols.append(x) for i, x in enumerate(cat_cols) if i in (range(42, 52))]
df[num_cols].dtypes

# ketegorik değişkenlerin içerisinde object olmayanların düşürülmesi
cat_cols = [x for i, x in enumerate(cat_cols) if i not in (range(42, 52))]
df[cat_cols].dtypes

# num_cols.append(list(cat_cols[42:52]))
# len(num_cols)
len(cat_cols)


# del num_cols[-10]

# cat_cols.append(num_cols[2])


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
# bu değişkenlerin %80'ninden fazlası null değerler olduğu için düşürdüm
df = df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

# çıkardıktan sonra değişkenleri tekrar yakaladım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Id ve SalePrice yeniden düşürdüm
num_cols = num_cols[1:-1]

# yukarıda değişkenleri düşürdüğüm için index'ler de değişiklik oldu
len(cat_cols)

[num_cols.append(x) for i, x in enumerate(cat_cols) if i in (range(38, 49))]
df[num_cols].dtypes

# ketegorik değişkenlerin içerisinde object olmayanların düşürülmesi
cat_cols = [x for i, x in enumerate(cat_cols) if i not in (range(38, 49))]
df[cat_cols].dtypes

len(cat_cols)

df[num_cols].dtypes

df[num_cols].isnull().sum()

# numeric değişkenleri en çok tekrar median ile doldurma
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

df["GarageYrBlt"]

df["FireplaceQu"]
df["MasVnrType"]
# na_values = ["FireplaceQu", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", \
#              "BsmtFinType1", "GarageCond", "GarageYrBlt", "GarageFinish", \
#              "GarageQual", "GarageType"]

# df[na_values].isnull().sum()
len(cat_cols)

df[cat_cols].isnull().sum() / df.shape[0]


def fill_na_values_with_mode(dataframe, columns):
    for col in columns:
        # print(dataframe[col])
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode().iloc[0])


# kategorik değişkenleri en çok tekrar eden değeriyle doldurma
fill_na_values_with_mode(df, cat_cols)

missing_values_table(df)


# df[cat_cols].dtypes
# df[cat_cols] = df[cat_cols].astype("object")


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


df = rare_encoder(df, 0.01)

df


# Adım 3: Yeni değişkenler oluşturunuz.

def set_type(dataframe, col, type):
    dataframe[col] = dataframe[col].astype(type)


# yapı yaşı
df["Age_Building"] = df["YearRemodAdd"] - df["YearBuilt"]

# ev kalitesi
set_type(df, 'OverallCond', float)
df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']

# toplam alan
df['TotalArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'] + + df['GarageArea']

# yaşam alanı oranı
df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalArea']

# toplam banyo sayısı
# set_type(df, ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"], int)
df['TotalBathrooms'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']

# garaj kapasitesi
df['GarageCapacity'] = df['GarageCars'] + df['GarageArea']

# Adım 4: Encoding işlemlerini gerçekleştiriniz.

# Label Encoding
binary_cols = [col for col in df.columns if (df[col].dtype not in [int, float]) and (df[col].nunique() == 2)]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

# One Hot Encoding
ohe_cols = [col for col in df.columns if 25 >= df[col].nunique() > 2]


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ohe_cols)

# scaler = StandardScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])

# Standartlaştırma

# df[num_cols] = scaler.fit_transform(df[num_cols])

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "SalePrice"]
num_cols = [col for col in num_cols if col not in "Id"]
num_cols.append(cat_cols[0])
df[num_cols].dtypes

df[cat_cols].dtypes

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Görev 3: Model Kurma

# Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)

df.reset_index(inplace=True, drop=True)
variables = [col for col in df.columns if col not in "SalePrice"]

X_train = df.loc[:1459, variables]
X_test = df.loc[1460:, variables]
y_train = df.loc[:1459, "SalePrice"]
y_test = df.loc[1460:, "SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25,
                                                  random_state=1)

# Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.

# Random Forest
rf_model = RandomForestRegressor(random_state=17).fit(X_train, y_train)

y_pred = rf_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 29513.102537211216

# GBM
gbm_model = GradientBoostingClassifier().fit(X_train, y_train)

y_pred = gbm_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 28462.53252809921

# LightGBM
lgbm_model = LGBMRegressor().fit(X_train, y_train)

y_pred = lgbm_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 28462.53252809921

# XGBoost
xgb_model = XGBRegressor().fit(X_train, y_train)

y_pred = xgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 31739.217551795515

# CatBoost
catb_model = CatBoostRegressor().fit(X_train, y_train)

y_pred = catb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 24205.191153001928

# Decision Tree
cart_model = DecisionTreeRegressor().fit(X_train, y_train)

y_pred = cart_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 40119.669753759


# Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.

# RandomForest
rf_params = {'max_depth': list(range(1, 7)),
             'max_features': [3, 5, 7],
             'n_estimators': [100, 200, 500, 1000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X_train, y_train)

y_pred = rf_final.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 40017.47684707109


# XGBoost
xgb_params = {'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'n_estimators': [100, 200, 500],
              'max_depth': [3, 4, 5, 6, 7],
              'learning_rate': [0.3, 0.5]}

xgb_best_grid = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

xgb_best_grid.best_params_

xgb_final = xgb_model.set_params(**xgb_best_grid.best_params_).fit(X_train, y_train)

y_pred = xgb_final.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 29908.65106567284


# LightGBM
lgbm_params = {'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
               'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
               'n_estimators': [100, 200, 500, 100],
               'max_depth': list(range(1, 7))}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X_train, y_train)

y_pred = lgbm_final.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 27961.611470086133


# CatBoost
catb_params = {'iterations': [200, 500, 1000],
               'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
               'depth': list(range(1, 8))}

catb_best_grid = GridSearchCV(catb_model, catb_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

catb_best_grid.best_params_

catb_model = CatBoostRegressor()

catb_final = catb_model.set_params(**catb_best_grid.best_params_).fit(X_train, y_train)

y_pred = catb_final.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


# 24478.61100087933

# Adım 4: Değişken önem düzeyini inceleyeniz.
# Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir
# dataframe oluşturup sonucunuzu yükleyiniz.

def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X_train)
plot_importance(xgb_final, X_train)
plot_importance(lgbm_final, X_train)
plot_importance(catb_final, X_train)


# Prediction

# RandomForest
rf_pred = rf_final.predict(X_train)
rf_pred_score = np.sqrt(mean_squared_error(y_train, rf_pred))
# 33116.99886321384

# XGBoost
xgb_pred = xgb_final.predict(X_train)
xgb_pred_score = np.sqrt(mean_squared_error(y_train, xgb_pred))
# 4770.08918830866

# LightGBM
lgbm_pred = lgbm_final.predict(X_train)
lgbm_pred_score = np.sqrt(mean_squared_error(y_train, lgbm_pred))
# 15604.81879637132

# CatBoost
catb_pred = catb_final.predict(X_train)
catb_pred_score = np.sqrt(mean_squared_error(y_train, catb_pred))
# 6122.062142230834

# tahmin edilen değerlerin kolay kıyaslanabilmesi açısından bir DataFrame oluşturdum
all_pred_score = pd.DataFrame(
    {"Model": ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
     "RMSE": [rf_pred_score, xgb_pred_score, lgbm_pred_score, catb_pred_score]},
    index=range(1, 5))

all_pred_score = all_pred_score.sort_values("RMSE", ascending=True).reset_index()
# del all_pred_score["index"]
all_pred_score

