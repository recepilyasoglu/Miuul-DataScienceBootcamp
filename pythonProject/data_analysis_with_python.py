# NumPy

import numpy as np
a = [1,2,3,4]
b = [2,3,4,5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

#with numpy
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a * b

import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10) #0 ile 10 arasında rastgele 10 integer
np.random.normal(10, 4, (3,4)) #ortalaması 10, standart sapması 4 olan, 3'e 4 array oluşturma

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

# reshaping
import numpy as np
np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)