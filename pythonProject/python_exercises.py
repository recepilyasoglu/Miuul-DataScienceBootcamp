import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Q: Capitalize all letters of the given string expression.
# Put space instead of comma, separate word by word.

text = "The goal is to turn data into information, and information into insight"
text = text.upper()
text = text.split(" ")

#Q2: Follow the steps below to the given list.
#Step1: Look at the number of elements of the given list.
#Step 2: Call the elements at index zero and ten.
#Step 3: Create a list ["D", "A", "T", "A"] from the given list.
#Step 4: Delete the element in the eighth index.
#Step 5: Add a new element.
#Step 6: Re-add element "N" to the eighth index.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)
print(lst[0], lst[10])
print(lst[0:4])
lst[8]
lst.remove("N")
lst.append("!")
lst[8].append("N")

#Q3: Apply the following steps to the given dictionary structure.
#Step 1: Access the key values.
#Step 2: Access the values.
#Step 3: Update the value 12 of the Daisy key to 13.
#Step 4: Add a new value whose key value is Ahmet value [Turkey,24].
#Step 5: Delete Antonio from dictionary.

dict = {'Cristian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

dict.keys()
dict.values()

d1 = {'Daisy': 13}
dict.update(d1)

d2 = {"Ahmet": ["Turkey", 24]}
dict.update(d2)

dict.pop("Antonio")
