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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV, \
    validation_curve
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
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

# Standartlaştırma

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

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice']  # np.log1p(df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          # ("Ridge", Ridge()),
          # ("Lasso", Lasso()),
          # ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          # ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
# ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

df['SalePrice'].mean()
df['SalePrice'].std()

##################
# BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.
# Not: Log'un tersini (inverse) almayı unutmayınız.
##################

# Log dönüşümünün gerçekleştirilmesi

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred

# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = np.expm1(y_pred)
new_y

new_y_test = np.expm1(y_test)
new_y_test

np.sqrt(mean_squared_error(new_y_test, new_y))

# RMSE : 22118.413146021652


########################################################
# Hiperparametre optimizasyonlarının gerçekleştirilmesi
########################################################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)


################################################################
# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")


model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X, 20)

##########################################################################################
# test dataframe'indeki boş olan salePrice değişkenlerini tahminleyiniz ve
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
##########################################################################################

model = LGBMRegressor()
model.fit(X, y)
predictions = model.predict(test_df.drop(["Id", "SalePrice"], axis=1))

dictionary = {"Id": test_df.index, "SalePrice": predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)

