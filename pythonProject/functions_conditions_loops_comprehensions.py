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
