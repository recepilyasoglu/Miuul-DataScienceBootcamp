def calculate(x):
    for i in range(x):
        i = i * 2
        print(i)


calculate(5)


def calculate2(x, y):
    print("Toplamları..: ", x + y)
    print("Farkları...: ", x - y)
    print("Çarpımları...: ", x * y)


calculate2(5, 3)


# Docstring
def calculate2(x, y):
    """
    Calculation of two numbers

    Args:
        x: int, float
        y: int, float

    Returns:
        int, float

    """
    print("Toplamları..: ", x + y)
    print("Farkları...: ", x - y)
    print("Çarpımları...: ", x * y)


calculate2(5, 3)

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 3)
add_element(5, 3)
add_element(180, 10)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output


# varm, moisture, charge, output = calculate(98, 22, 78)


def standartdization(a, p):
    return a * 10 / 100 * p * p


standartdization(45, 1)


def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standartdization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 19, 12)


# Uygulama - Mülakat Sorusu
# Amaç: Aşağıdaki şekilde değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i am LeArNiNg pYtHoN"

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)


alternating("hi my name is john and i am learning python")

# enumerate
students = ["John", "Mark", "Vanessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

# Öğrenci indexlerine göre tek-çift sıralama
A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

    print(index, student)

# Uygulama

# divide_students fonksiyonu yaz
# çift indexte yer alan öğrencileri bir listeye alınız.
# tek indexte yer alan öğrenciler başka bir listeye alınacak.
# fakat bu iki liste tek bir liste olarak return olsun

students = ["Johns", "Mark", "Vanessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


divide_students(students)


# alternating fonksiyonunun enumerate ile yazılması


def alter_enum(text):
    new_text = ""
    for i, letter in enumerate(text):
        if i % 2 == 0:
            new_text += letter.upper()
        else:
            new_text += letter.lower()
    print(new_text)


alter_enum("hi my name is john and i am learning python")

# lambda, map, filter, reduce
new_sum = lambda a, b: a + b

new_sum(4, 5)

salaries = [1000, 2000, 3000, 4000, 5000]


# map
def new_salary(x):
    return x * 20


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

# use map
list(map(new_salary, salaries))

list(map(lambda x: x * 20 / 100 + x, salaries))

list(map(lambda x: x ** 2, salaries))

# Filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# Reduce
from functools import reduce

list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)

# COMPREHENSIONS

## List Comprehensions

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

[new_salary(salary * 2) if salary > 3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Vanessa", "Mariam"]

students_no = ["John", "Vanessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

# Dict Comprehensions
dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4
              }

dictionary.keys()
dictionary.values()
dictionary.items()
# square of values
{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}

{k.upper(): v * 2 for (k, v) in dictionary.items()}

# Uygulama: Çift sayıalrın karesi alınarak bir sözlüğe eklenmek istenmektedir
# Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak

numbers = range(10)
new_dict = {}

for i in numbers:
    if i % 2 == 0:
        new_dict[i] = i ** 2

{i: i ** 1 for i in numbers if i % 2 == 0}

# Applications of List & Dict Comprehensions
## Uygulama: Bir veri setindeki değişken isimlerini değiştirmek

# before : ["total","speeding","alcohol","not_distracted","no_previous","ins_premium","in_losses","abbrev"]
# after : ["TOTAL","SPEEDİNG","ALCOHOL","NOT_DİSTRACTED","NO_PREVİOUS","İNS_PREMİUM","İNS_LOSSES","ABBREV"]

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

# with List Comprehensions
df = sns.load_dataset("car_crashes")
df.columns
A = []
df.columns = [col.upper() for col in df.columns]

# Uygulama2: İsminde "INS" olan değikenlerin başına FLAG diğerlerine NO_FLAG eklemek

# step by step
[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

# Uygulama3: Key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak
# Sadece sayısal değikenler için yapmak istiyoruz

# {"total" : ["mean", "min", "max", "var"],
# "speeding" : ["mean", "min", "max", "var"],
# "alcohol" : ["mean", "min", "max", "var"],
# "not_distracted" : ["mean", "min", "max", "var"],
# "no_previous" : ["mean", "min", "max", "var"],
# "ins_premium" : ["mean", "min", "max", "var"],
# "ins_losses" : ["mean", "min", "max", "var"],
# }

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

# veri seti içerisindeki object olmayan değerleri al(sayısal değerleri al)
num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

#with Comprehensions
new_dict = {col: agg_list for col in num_cols}

df[num_cols].agg(new_dict)
