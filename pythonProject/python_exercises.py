import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Q2: Capitalize all letters of the given string expression.
# Put space instead of comma, separate word by word.

text = "The goal is to turn data into information, and information into insight"
text = text.upper()
text = text.split(" ")

# Q3: Follow the steps below to the given list.
# Step1: Look at the number of elements of the given list.
# Step 2: Call the elements at index zero and ten.
# Step 3: Create a list ["D", "A", "T", "A"] from the given list.
# Step 4: Delete the element in the eighth index.
# Step 5: Add a new element.
# Step 6: Re-add element "N" to the eighth index.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)
print(lst[0], lst[10])
print(lst[0:4])
lst[8]
lst.remove("N")
lst.append("!")
lst[8].append("N")

# Q4: Apply the following steps to the given dictionary structure.
# Step 1: Access the key values.
# Step 2: Access the values.
# Step 3: Update the value 12 of the Daisy key to 13.
# Step 4: Add a new value whose key value is Ahmet value [Turkey,24].
# Step 5: Delete Antonio from dictionary.

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

# Q5: Taking a list as an argument, assigning the odd and even numbers in the list to separate lists,
# and adding these lists. Write a function that returns.

l = [2, 13, 18, 93, 22]


def check_func(l):
    even_list = []
    odd_list = []
    for i in l:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

even_list, odd_list = check_func(l)

even_list
odd_list

#Q6: The names of the students who received degrees in engineering and medicine faculties are listed below.
# exists. Respectively, the first three students represent the success order of the engineering faculty, while the last three students
# belongs to the medical faculty student list. Print student degrees specific to faculty using Enumarate

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
tip = ogrenciler[3:]
muhendislik = ogrenciler[:3]

for index, engineer in enumerate(muhendislik, start=1):
        print("Mühendislik Fakültesi", index, ".", "öğrenci: ", engineer)

for index, doctor in enumerate(tip, start=1):
        print("Tıp Fakültesi", index, ".", "öğrenci: ", doctor)


