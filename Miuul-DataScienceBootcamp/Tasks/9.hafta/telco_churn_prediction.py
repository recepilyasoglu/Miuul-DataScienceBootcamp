###########################################
############# RECEP İLYASOĞLU #############
###########################################

############ Telco Churn Prediction ############

# İş Problemi

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.


# Veri Seti Hikayesi

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayanhayalibir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunug österir

# Importing Libraries
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

# df.corr().sort_values("Churn", ascending=False) \
#     .drop("Churn", axis=0)

# Adım 2: Yeni değişkenler oluşturunuz.

df.columns

# df["Num_Gender"] = [0 if col == "Male" else 1 for col in df.gender]
# df[["gender", "Num_Gender"]].head(20)

df["Cat_Tenure"] = pd.cut(df["tenure"], bins=[0, 10, 15, 72], labels=["New", "Star", "Loyal"])
df[["Cat_Tenure", "tenure"]].head(20)

df["MonthlyCharges"].describe().T,

df["Cat_MonthlyCharges"] = pd.cut(df["MonthlyCharges"], bins=[df.MonthlyCharges.min(), 40, 70, df.MonthlyCharges.max()],
                                  labels=["Lower", "Middle", "High"], right=False)

df[["MonthlyCharges", "Cat_MonthlyCharges"]].head(20)
df["Cat_MonthlyCharges"].value_counts()

df.groupby(["Cat_Tenure", "PaymentMethod"]).agg({"PaymentMethod": "count"})

df["ContractLength"] = np.where(df["Contract"] == "Month-to-month", "Short", "Long")
df[["Contract", "ContractLength"]].head(20)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
new_variables = df[["Cat_Tenure", "Cat_MonthlyCharges", "ContractLength"]]


def count_of_values(dataframe):
    for col in dataframe:
        print("#######", col, "Değişkeninin Değer Sayısı #######", "\n",
              dataframe[col].value_counts())


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

random_user = X.sample(1)  # rastgele bir kullanıcı oluşturuyoruz

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
# 0.8053167408234181
gbm_f1 = gbm_cv_results['test_f1'].mean()
# 0.5916996422107582
gbm_auc = gbm_cv_results['test_roc_auc'].mean()
# 0.84598827585678

# LightGBM
lgbm_model = LGBMClassifier().fit(X, y)

lgbm_cv_results = cross_validate(lgbm_model,
                                 X, y,
                                 cv=5,
                                 scoring=["accuracy", "f1", "roc_auc"])

lgbm_test = lgbm_cv_results['test_accuracy'].mean()
# 0.7950782563508408
lgbm_f1 = lgbm_cv_results['test_f1'].mean()
# 0.5756609171367744
lgbm_auc = lgbm_cv_results['test_roc_auc'].mean()
# 0.8350053407214943
lgbm_model.predict(random_user)

# XGBoost
xgboost_model = XGBClassifier(use_label_encoder=False)

xg_cv_results = cross_validate(xgboost_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

xg_test = xg_cv_results['test_accuracy'].mean()
# 0.7839858824147905
xg_f1 = xg_cv_results['test_f1'].mean()
# 0.5597840395783735
xg_auc = xg_cv_results['test_roc_auc'].mean()
# 0.8256516811522634


# K-NN
knn_model = KNeighborsClassifier().fit(X, y)

# cross validation
knn_cv_results = cross_validate(knn_model,
                                X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

knn_test = knn_cv_results['test_accuracy'].mean()
# 0.7519915156992926
knn_f1 = knn_cv_results['test_f1'].mean()
# 0.4495756497728191
knn_auc = knn_cv_results['test_roc_auc'].mean()
# 0.7040636259728742
knn_model.predict(random_user)

# Decision Tree
dt_model = DecisionTreeClassifier().fit(X, y)

dt_cv_results = cross_validate(dt_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

dt_test = dt_cv_results['test_accuracy'].mean()
# 0.7296623972193493
dt_f1 = dt_cv_results['test_f1'].mean()
# 0.495272831796155
dt_auc = dt_cv_results['test_roc_auc'].mean()
# 0.6571461457034756
dt_model.predict(random_user)

best_model_results = pd.DataFrame(
    {"Model": ["Logistic Regression", "Random Forest", "GBM", "LightGBM", "XGBoost", "KNN", "Decision Tree"],
     "Accuracy": [log_test, rf_test, gbm_test, lgbm_test, xg_test, knn_test, dt_test],
     "AUC": [log_auc, rf_auc, gbm_auc, lgbm_auc, xg_auc, knn_auc, dt_auc],
     "F1_Score": [log_f1, rf_auc, gbm_auc, lgbm_auc, xg_auc, knn_auc, dt_auc]},
    index=range(1, 8))

best_model_results = best_model_results.sort_values("Accuracy", ascending=False)

top4_models = best_model_results.head(4).reset_index()
# del top4_models["index"]


# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin
# ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.

# Seçilen Modeller: GBM, Logistic Regression, LightGBM, Random Forest

# GBM
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}  # kaç tane gözlemin oransal olarak göz önünde bulundurulacağını ifade eder

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

gbm_final_cv_results = cross_validate(gbm_final,
                                      X, y,
                                      cv=5,
                                      scoring=["accuracy", "f1", "roc_auc"])

gbm_final_test = gbm_final_cv_results['test_accuracy'].mean()
# 0.8018179193319119 -> 0.8054588872342212
gbm_final_f1 = gbm_final_cv_results['test_f1'].mean()
# 0.582522324396216 -> 0.592108885371575
gbm_final_auc = gbm_final_cv_results['test_roc_auc'].mean()
# 0.8455268626584862 -> 0.8460296805268179


# Logistic Regression
log_model.get_params()

log_params = {"penalty": ['l1', 'l2'],
              'C': np.logspace(-3, 3, 7),
              "solver": ['newton-cg', 'lbfgs', 'liblinear']}

log_best_grid = GridSearchCV(log_model, log_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

log_best_grid.best_params_

log_final = log_model.set_params(**log_best_grid.best_params_, random_state=17).fit(X, y)

log_final_cv_results = cross_validate(log_final,
                                      X, y,
                                      cv=5,
                                      scoring=["accuracy", "f1", "roc_auc"])

log_final_test = log_final_cv_results['test_accuracy'].mean()
# 0.8059382470763069 -> 0.8057445954539435
log_final_f1 = log_final_cv_results['test_f1'].mean()
# 0.5911821068005934 -> 0.5913753619751492
log_final_auc = log_final_cv_results['test_roc_auc'].mean()
# 0.8463000059038415 -> 0.8458423917254893

# LightGBM
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

lgbm_final_cv_results = cross_validate(lgbm_final,
                                       X, y,
                                       cv=5,
                                       scoring=["accuracy", "f1", "roc_auc"])

lgbm_final_test = lgbm_final_cv_results['test_accuracy'].mean()
# 0.7938640805711701 -> 0.804321918147527
lgbm_final_f1 = lgbm_final_cv_results['test_f1'].mean()
# 0.574774214849072 -> 0.5878492992727085
lgbm_final_auc = lgbm_final_cv_results['test_roc_auc'].mean()
# 0.833585885238031 -> 0.8452201168774376


# RandomForest
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

rf_final_cv_results = cross_validate(rf_final,
                                     X, y,
                                     cv=5,
                                     scoring=["accuracy", "f1", "roc_auc"])

rf_final_test = rf_final_cv_results['test_accuracy'].mean()
# 0.7910226666989727 -> 0.8030412861520482
rf_final_f1 = rf_final_cv_results['test_f1'].mean()
# 0.5533446100583161 -> 0.5768646484342799
rf_final_auc = rf_final_cv_results['test_roc_auc'].mean()
# 0.8244377769644089 -> 0.8465187815450934
rf_model.predict(random_user)


# Hiperparametre optimizasyonlarından elde edilen sonuçlar ve önceki sonuçların kıyaslanması

top4_models["New_Accuracy"] = [log_final_test, gbm_final_test, lgbm_final_test, rf_final_test]
top4_models["New_AUC"] = [log_final_auc, gbm_final_auc, lgbm_final_auc, rf_final_auc]
top4_models["New_F1_Score"] = [log_final_f1, gbm_final_f1, lgbm_final_f1, rf_final_f1]

top4_models.sort_values("New_Accuracy", ascending=False)


# Features Importing
def plot_importance(model, features, num=len(X), save=False):
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

# plot_importance(log_final, X)
plot_importance(gbm_final, X)
plot_importance(lgbm_final, X)
plot_importance(rf_final, X)
