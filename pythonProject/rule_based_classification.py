import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Calculating Lead-Based Returns with Rule-Based Classification

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


# Question 7: What are the sales numbers by SOURCE types?

# Question 8: What are the PRICE averages by country?

# Question 9: What are the PRICE averages according to SOURCEs?

# Question 10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?