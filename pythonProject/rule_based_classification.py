import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Calculating Lead-Based Returns with Rule-Based Classification

## Task 1: Answer the Following Questions
# Question 1: Read the persona.csv file and show the general information about the dataset.
data = pd.read_csv("persona.csv")
df = data.copy()
df.head()
df.info()
df.describe().T
df.isnull().sum()

# Question 2: How many unique SOURCE are there? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Question 3: How many unique PRICEs are there?
df["PRICE"].nunique()

# Question 4: How many sales were made from which PRICE?
df["PRICE"].value_counts()

# Question 5: How many sales were made from which country?
df["COUNTRY"].value_counts()

# Question 6: How much was earned in total from sales by country?
df.groupby("COUNTRY").agg({"PRICE": "sum"}).sort_values("PRICE", ascending=False)

# Question 7: What are the sales numbers by SOURCE types?
df.groupby("SOURCE").agg({"PRICE": "sum"})

# Question 8: What are the PRICE averages by country?
df.groupby("COUNTRY").agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# Question 9: What are the PRICE averages according to SOURCE's?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Question 10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

## Task 2: What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

## Task 3: Sort the output according to PRICE.

# To better see the output in the previous question, apply the sort_values method in descending order of PRICE.
# Save the output as agg_df.
agg_df = agg_df.sort_values("PRICE", ascending=False)

## Task 4: Convert the names in the index to variable names
agg_df.reset_index(inplace=True)

## Task 5: Convert age variable to categorical variable and add it to agg_df.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70],
                           labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
agg_df

## Task 6: Identify new level-based customers (personas).
# Define new level-based customers (personas) and add them as variables to the dataset.
# Name of the new variable to be added: customers_level_based
# You need to create the customers_level_based variable by combining the observations from the output from the previous question.

agg_df["customer_level_based"] = [(agg_df.COUNTRY[col] + "_" + agg_df.SOURCE[col] + "_" +
                                   agg_df.SEX[col] + "_" + agg_df.AGE_CAT[col]).upper() for col in agg_df.index]

agg_df.loc[:, ["customer_level_based", "PRICE"]].sort_values("PRICE", ascending=False)

agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})

## Task 7: Segment new customers (personas).
# Divide new customers (Example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Add the segments to agg_df as a variable with the SEGMENT naming.
# Describe segments (group by segments and get price mean, max, sum)

agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"].rank(method="first"), 4, labels=["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

## Task 8: Categorize new customers and estimate how much revenue they can generate.
# What segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to earn on average?
# What segment does a 35-year-old French woman using IOS belong to and how much income is expected to earn on average?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user]

agg_df.loc[agg_df["customer_level_based"] == new_user, ["PRICE", "SEGMENT"]].agg({"PRICE": "mean"})

new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user2]

agg_df.loc[agg_df["customer_level_based"] == new_user2, ["PRICE", "SEGMENT"]].agg({"PRICE": "mean"})
