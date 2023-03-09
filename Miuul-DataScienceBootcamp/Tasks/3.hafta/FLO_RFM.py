################ RECEP İLYASOĞLU ################

###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
import pandas as pd
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option("display.width", 500)

# 1. flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv(r"Tasks/3.hafta/flo_data_20k.csv")
df = df_.copy()

# 2. Veri setinde
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boş değer,
df.isnull().sum()
# e. Değişken tipleri, incelemesi yapınız.
df.dtypes

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_number_purchase"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_number_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
date = df.columns[df.columns.str.contains("date")]
df[date] = df[date].apply(pd.to_datetime)
df.dtypes

# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel")["total_number_purchase", "total_number_price"].agg({"count", "mean"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df["total_number_price"].sort_values(ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df["total_number_purchase"].sort_values(ascending=False).head(10)


# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_preparation(dataframe):
    dataframe["total_number_purchase"] = dataframe["order_num_total_ever_offline"] + dataframe[
        "order_num_total_ever_online"]
    dataframe["total_number_price"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    dt = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[dt] = dataframe[dt].apply(pd.to_datetime)

    return dataframe


df = df_.copy()
new_df = data_preparation(df)
new_df

# GÖREV 2: RFM Metriklerinin Hesaplanması
# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                   "total_number_purchase": lambda x: x,
                                   "total_number_price": lambda x: x})
rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T


# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                   rfm["frequency_score"].astype(str))

rfm = rfm[(rfm['frequency'] > 1)]


rfm.describe().T

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "need_attention",
    r"33": "about_to_sleep",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()

# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
# ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
# yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
main_df = pd.merge(df, rfm, on="master_id", how="inner")

tsk_df = main_df[(main_df["interested_in_categories_12"].str.contains("KADIN") & ((main_df["segment"] == "champions") | (main_df["segment"] == "loyal_customers")))]
tsk_df.master_id.to_csv("special_customers.csv")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
main_df.segment.head(35)
tsk_df2 = main_df[(main_df["interested_in_categories_12"].str.contains("ERKEK")) & (main_df["interested_in_categories_12"].str.contains("COCUK") & ((main_df["segment"] == "cant_loose") | (main_df["segment"] == "hibernating") | (main_df["segment"] == "new_customers")))]
tsk_df2.head()
tsk_df2.master_id.to_csv("male-child_customers.csv")


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def rfm(dataframe, csv=False):
    # Verinin Hazırlanması
    dataframe["total_number_purchase"] = dataframe["order_num_total_ever_offline"] + dataframe["order_num_total_ever_online"]
    dataframe["total_number_price"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date] = dataframe[date].apply(pd.to_datetime)
    # RFM Metriklerinin Hazırlanması
    today_date = dt.datetime(2021, 6, 1)
    rfm = df.groupby("master_id").agg(
        {"last_order_date": lambda x: (today_date - x.max()).days,
         "total_number_purchase": lambda x: x,
         "total_number_price": lambda x: x})
    rfm.columns = ["recency", "frequency", "monetary"]
    # RF ve RFM Skorlarının Oluşturulması
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) +
                       rfm["frequency_score"].astype(str))
    # RF Skorlarının Segment Olarak Tanımlanması
    seg_map = {
        r"[1-2][1-2]": "hibernating",
        r"[1-2][3-4]": "at_Risk",
        r"[1-2]5": "cant_loose",
        r"3[1-2]": "need_attention",
        r"33": "about_to_sleep",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions",
    }
    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

df = df_.copy()

new_df2 = rfm(df, csv=True)
