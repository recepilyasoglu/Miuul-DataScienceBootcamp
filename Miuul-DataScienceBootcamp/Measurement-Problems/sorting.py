### Sorting Products

# Uygulama: Kurs Sıralama
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("Measurement-Problems/product_sorting.csv")
df.head()
df.shape


# Sorting by Rating
df.sort_values("rating", ascending=False).head(20)





