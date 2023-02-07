#Q1: Capitalize the names of the numeric variables in the car_crashes data using the List Comprehension structure.
#Convert it to letter and add NUM at the beginning.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

["NUM_"+col.upper() if df[col].dtype in ["int64", "float64"] else col.upper() for col in df.columns]

