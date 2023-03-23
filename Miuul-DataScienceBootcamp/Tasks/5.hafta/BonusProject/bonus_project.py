################ RECEP İLYASOĞLU ################
################ BONUS PROJECT ################
### Association Rule Based Recommender System

## Task 1: Preparing the Data
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Step 1: Read the 2010-2011 sheet from the Online Retail II dataset.
df_ = pd.read_excel("Tasks/5.hafta/BonusProject/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape
df.describe().T

# Step 2: Drop the observation units whose StockCode is POST. (POST price added to each invoice does not represent the product.)

df = df[df["Country"] == "Germany"]

df = df[~df["StockCode"].str.contains("POST", na=False)]

# Step 3: Drop the observation units with null values.

df.isnull().sum()
df.dropna(inplace=True)

# Step 4: Extract the values with C in Invoice from the data set. (C means the cancellation of the invoice.)

df = df[~df["StockCode"].str.contains("C", na=False)]
df.shape

# Step 5: Filter out the observation units whose price is less than zero.
df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]  # - li değerleden kurtulduk ama aykırı değerler mevcut


# Step 6: Examine the outliers of the Price and Quantity variables, suppress if necessary
def outlier_thresholds(dataframe, variable):  # Amacı: kendisine girilen değişken için eşik değer belirlemektir
    quartile1 = dataframe[variable].quantile(0.01)  # quantile: çeyreklik hesaplama için
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):  #
    low_limit, upl_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > upl_limit), variable] = upl_limit


replace_with_threshold(df, "Quantity")
replace_with_threshold(df, "Price")

## Task 2: Generating Association Rules Through German Customers

# Step 1: Define the create_invoice_product_df function that will create the invoice product pivot table as follows.
df.head()


def create_invoice_product_df(dataframe):
    return dataframe.groupby(["Invoice", "Description"])["Description"].count() \
            .unstack() \
            .fillna(0) \
            .applymap(lambda x: 1 if x > 0 else 0)

pivot_df = create_invoice_product_df(df)

# Step 2: Define the create_rules function that will create the rules and find the rules for german customers
frequent_itemsets = apriori(pivot_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules["support"].describe().T
rules["confidence"].describe().T
rules["lift"].describe().T

rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.1) & (rules["lift"] > 7)]. \
    sort_values("confidence", ascending=False)


# Task 2: Making Product Suggestions to Users Given the Product IDs in the Basket

# Step 1: Find the names of the given products using the check_id function.


# Step 2: Make a product recommendation for 3 users using the arl_recommender function.


# Step 3: Look at the names of the products to be recommended.


