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


