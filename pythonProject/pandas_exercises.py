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

# Task 11: Find the sum of the null values in each variable.
df.isnull().sum()

# Task 12: Extract the variable who from the dataframe.
df.drop(columns="who", inplace=True)
df.columns

# Task 13: Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.
df["deck"] = df["deck"].fillna(df["deck"].mode())

# Task 14: Fill in the blank values in the age variable with the median of the age variable.
df["age"] = df["age"].fillna(df["age"].median())

# Task 15: Find the sum, count, mean values of the pclass and gender variables of the survived variable.

# Task 16: Write a function that returns 1 for those under 30 and 0 for those above or equal to 30. titanic data using the function you wrote
#create a variable named age_flag in the set. (use apply and lambda constructs)

# Task 17: Define the Tips dataset from the Seaborn library.
df2 = sns.load_dataset("Tips")
df2.head()

# Task 18: Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the Time variable.

# Task 19: Find the sum, min, max and average of total_bill values by days and time.
