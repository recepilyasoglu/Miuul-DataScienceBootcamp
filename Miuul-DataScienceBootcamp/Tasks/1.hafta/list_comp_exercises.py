####### RECEP İLYASOĞLU #######
###### List Comprehension Exercises ######

#Q1: Capitalize the names of the numeric variables in the car_crashes data using the List Comprehension structure.
#Convert it to letter and add NUM at the beginning.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

["NUM_"+col.upper() if df[col].dtype in ["int64", "float64"] else col.upper() for col in df.columns]


#Q2: Using the List Comprehension structure, the car_crashes data does not contain "no" in its name.
# Write "FLAG" after the names of the variables.

[col.upper()+"_FLAG" if "no" not in col else col.upper() for col in df.columns]

#Q3 = Different from the variable names given below using the List Comprehension structure.
#select the names of the variables and create a new dataframe

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]

new_df.head()

