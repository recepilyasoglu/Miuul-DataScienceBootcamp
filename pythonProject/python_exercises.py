####### RECEP İLYASOĞLU #######
###### Python Exercises ######

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Q1: Examine the data structures of the given values.

x = 8
type(x)
y = 3.2
type(x+y)
type(y)
z = 8j+18
type(z)
a = "Hello World"
type(a)
b = True
type(b)
c = 23 < 22
type(c)
l = [1, 2, 3, 4]
type(l)
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)
t = {"Machine Learning", "Data Science"}
type(t)
s = {"Python", "Machine Learning", "Data Science"}
type(s)


# Q2: Capitalize all letters of the given string expression.
# Put space instead of comma, separate word by word.

text = "The goal is to turn data into information, and information into insight"
(text.upper()).split(" ")
text = text.upper()
text = text.replace(",", " ")
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
lst.pop(8)
lst.append("!")
lst.insert(8, "N")


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


# Q6: The names of the students who received degrees in engineering and medicine faculties are listed below.
# exists. Respectively, the first three students represent the success order of the engineering faculty, while the last three students
# belongs to the medical faculty student list. Print student degrees specific to faculty using Enumarate

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]
tip = ogrenciler[3:]
muhendislik = ogrenciler[:3]

for index, engineer in enumerate(muhendislik, start=1):
    print("Mühendislik Fakültesi", index, ".", "öğrenci: ", engineer)

for index, doctor in enumerate(tip, start=1):
    print("Tıp Fakültesi", index, ".", "öğrenci: ", doctor)


# Q7: Three lists are given below. The lists contain the code, credit and quota information of a course, respectively takes.
# Print course information using zip

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} notu olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir")


# Q8 = Below are 2 sets. If the 1st cluster includes the 2nd cluster, you are asked to specify their common elements.
# if not, you are expected to define the function that will print the difference of the 2nd set from the 1st set.

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def check_lists(l1, l2):
    if l1.issuperset(l2):
        print(l2.intersection(l1))
    else:
        print(l2.difference(l1))

check_lists(kume1, kume2)
