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




