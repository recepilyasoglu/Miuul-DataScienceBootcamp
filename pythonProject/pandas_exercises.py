# Task 1: Identify the Titanic dataset from the Seaborn library.
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option("display.max_columns", None)

df = sns.load_dataset("titanic")
df.head()

# Task 2: Find the number of male and female passengers in the Titanic dataset.
df["sex"].value_counts()

# Task 3: Find the number of unique values for each column.
df.nunique()

# Task 4: Find the number of unique values of the variable pclass.
df["pclass"].nunique()

# Task 5: Find the number of unique values of pclass and parch variables.
df[["pclass", "parch"]].nunique()

# Task 6: Check the type of the embarked variable. Change its type to category and check again.
df["embarked"].dtypes
df["embarked"].astype("category")

# Task 7: Show all the sages of those with embarked value C.
df[(df["embarked"] == "C")]

# Task 8: Show all the sages of those whose embarked value is not S.
df[(df["embarked"] != "S")]

# Task 9: Show all information for female passengers younger than 30 years old.
df[(df["age"] < 30) & (df['sex'] == "female")]

# Task 10: Show the information of passengers whose Fare is over 500 or 70 years old
df[(df["fare"] > 500) | (df["age"] > 70)]
