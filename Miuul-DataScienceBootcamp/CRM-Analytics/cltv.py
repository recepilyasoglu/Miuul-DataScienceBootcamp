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


## 3. Purchase Frequency (Total Transaction / Total Number of Customers)
cltv_c.head()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

cltv_c.shape[0]  # = TotalNumberofCustomers


## 4. Repeat Rate & Churn Rate (Birden fazla alışveriş yapan müşteriler sayısı / tüm müşteriler)

cltv_c[cltv_c["total_transaction"] > 1].shape[0]  # 1 den fazla alışveriş yapan müşterilerin sayısı geldi

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate


## 5. Profit Margin = Total Price * 0.10

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10


## 6. Customer Value = Average Order Value * Purchase Frequency

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]


## 7. Customer Lifetime Value (CLTV = (Customer Value / Churn Rate) x Profit Margin)

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending=False).head()


## 8. Creating Segments (Segmentlerin Oluşturulması)
# Müşterilere teker teker yaklaşım uygulamaktansa segmentlere ayırıp segment bazlı yaklaşım sergilemek mantıklı

cltv_c.sort_values(by="cltv", ascending=False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_c.sort_values(by="cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv_c.csv")


# 9. BONUS: Functionalization of the whole process (Tüm İşlemlerin Fonksiyonlaştırılması)

def create_cltv_c(dataframe, profit=0.10):

    # Veriyi Hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                                   "Quantity": lambda x: x.sum(),
                                                   "TotalPrice": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]
    # avg_order_value
    cltv_c["avg_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    #purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10
    # Customer Value
    cltv_c["customer_value"] = cltv_c["avg_order_value"] * cltv_c["purchase_frequency"]
    # Customer Lifetime Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    cltv_c.sort_values(by="cltv", ascending=False).head()
    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

df = df_.copy()

clv = create_cltv_c(df)
