#######################################################
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri) #
#######################################################

# 1. Veri Hazırlama
# 2. Average Order Value (Total Price / Total Transaction)
# 3. Purchase Frequency (Total Transaction / Total Number of Customers)
# 4. Repeat Rate & Churn Rate (Birden fazla alışveriş yapan müşteriler sayısı / tüm müşteriler)
# 5. Profit Margin = Total Price * 0.10
# 6. Customer Value = Average Order Value * Purchase Frequency
# 7. CLTV = (Customer Value / Churn Rate) x Profit Margin
# 8. Segmentlerin Oluşturulması
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması


## 1. Veri Hazırlama

## Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numaraç C ile başlıyorsa iptal edilen işlem
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihih ve zamanı
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz Müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # 0 dan sonra kaç basamak okuyacağını belirleme

df_ = pd.read_excel(r"CRM-Analytics/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]  # iptal edilen ürünleri düşürme
df.describe().T

df = df[(df["Quantity"] > 0)]  # Quantitiy min değeri 1 oldu, eksilerden kurtulduk
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]   # !!! Bu TotalPrice CLTV deki değil o müşteri bazlı, eğer groupby yapıp sum'ını alrısak o zaman olur

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),  # Her bir müşterinin eşsiz kaç tane faturası olduğunu görürüz bu da = Total Transaction
                                        "Quantity": lambda x: x.sum(),     # Bu tamamen analiz yapmak için, birinci önceliğimiz değil, önceliğimiz Invoice ve TotalPrice
                                        "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]


## 2. Average Order Value (Total Price / Total Transaction)

cltv_c.head()

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]




