###############################################
############### RECEP İLYASOĞLU ###############
###############################################

# Time Series - Iyzico - Task

# Business Problem

# Iyzico is a platform that facilitates the online shopping experience for both buyers and sellers.
# is a financial technology company. For e-commerce companies, marketplaces and individual users
# provides payment infrastructure. On merchant_id and day basis for the last 3 months of 2020
# Total trading volume is expected to be estimated.

import matplotlib
matplotlib.use("Qt5Agg")
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Task 1: Data Set Exploration

# Step 1: Read the Iyzico_data.csv file. Change the type of transaction_date variable to date.

data = pd.read_csv("Tasks/TimeSeries-iyzico/iyzico_data.csv", index_col=0)
df = data.copy()
df.head()

check_df(df)

df["transaction_date"] = df["transaction_date"].apply(pd.to_datetime)
df.dtypes

# Step 2: What are the start and end dates of the dataset?
df["transaction_date"].min(), df["transaction_date"].max()

# Step 3: What is the number of transactions in each merchant?
df.head()

df.groupby("merchant_id").agg({"Total_Transaction": ["sum", "mean", "count"]})

df.groupby("merchant_id")["Total_Transaction"].sum().sort_values(ascending=False)

# Step 4: What is the total payment amount in each merchant?
df.head()

df.groupby("merchant_id")["Total_Paid"].sum().sort_values(ascending=False)
df.groupby("merchant_id").agg({"Total_Paid": ["sum", "mean", "count"]})

# Step 5: Observe the transaction count graphs of the merchant in each year?
for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title = str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) & ( df.transaction_date >= "2018-01-01" ) & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2,title = str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) &( df.transaction_date >= "2019-01-01" )& (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()


# Task 2: Apply Feature Engineering techniques. Derive new features.

# • Date Features

def create_date_features(dataframe, date_col):
    dataframe['month'] = dataframe[date_col].dt.month
    dataframe['day_of_month'] = dataframe[date_col].dt.day
    dataframe['day_of_year'] = dataframe[date_col].dt.dayofyear
    dataframe['week_of_year'] = dataframe[date_col].dt.weekofyear
    dataframe['day_of_week'] = dataframe[date_col].dt.dayofweek
    dataframe['year'] = dataframe[date_col].dt.year
    dataframe["is_wknd"] = dataframe[date_col].dt.weekday // 4
    dataframe['is_month_start'] = dataframe[date_col].dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe[date_col].dt.is_month_end.astype(int)
    return dataframe

df = create_date_features(df, "transaction_date")
df.head()


# • Lag/Shifted Features
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

df.groupby(["Total_Transaction"]).agg({'Total_Paid': ["mean", "sum"]})

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby("merchant_id")["Total_Transaction"].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df)

df.head()

# • Rolling Mean Features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")["Total_Transaction"]. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])
check_df(df)


# • Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")["Total_Transaction"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)


# • Special days, exchange rate, etc.

########################
# Black Friday - Summer Solstice
########################

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]) ,"is_black_friday"]=1

df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]) ,"is_summer_solstice"]=1


# Task 3: Preparation and Modeling for Modeling

# Step 1: Do one-hot encoding.

df = pd.get_dummies(df, columns=["merchant_id", 'day_of_week', 'month'])

df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

check_df(df)


# Step 2: Define the Custom Cost Functions.
def smape(preds, target, epsilon=1e-10):
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)

    # Filter small values to avoid division by zero
    small_val_mask = denom < epsilon
    denom[small_val_mask] = epsilon

    smape_val = 200 * np.mean(num / denom)
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()  # LigtGBM veri yapısının içerisinde olan bağımlı değişkeni ifade ediyor
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# Step 3: Separate the dataset into train and validation.
# 2020'nin 10.ayına kadar train seti.
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

# 2020'nin son 3 ayı validasyon seti.
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', 'merchant_id', "Total_Transaction", "Total_Paid", "year"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

Y_train.isnull().any(), Y_val.isnull().any()


# Step 4: Create the LightGBM Model and observe the error value with SMAPE

lgb_params = {'metric': 'mae',
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbosity': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(params=lgb_params,
                  train_set=lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

# Plot Importance
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)

