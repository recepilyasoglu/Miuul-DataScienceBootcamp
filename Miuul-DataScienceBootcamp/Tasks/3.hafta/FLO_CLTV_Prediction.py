################ RECEP İLYASOĞLU ################
##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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
# GÖREV 1: Veriyi Hazırlama
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option("display.width", 500)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv(r"Tasks/3.hafta/flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):  # Amacı: kendisine girilen değişken için eşik değer belirlemektir
    quartile1 = dataframe[variable].quantile(0.01)  # quantile: çeyreklik hesaplama için
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):  #
    low_limit, upl_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > upl_limit), variable] = upl_limit


df.describe([0.1, 0.5, 0.99]).T

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

values = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]

for x in values:
    replace_with_threshold(df, x)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_number_purchase"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_number_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date = df.columns[df.columns.str.contains("date")]
df[date] = df[date].apply(pd.to_datetime)
df.dtypes

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
today_date = dt.datetime(2021, 6, 1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7

cltv_df["frequency"] = df["total_number_purchase"]
cltv_df["monetary_cltv_avg"] = df["total_number_price"] / df["total_number_purchase"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df.head()
df.head()

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(30)
cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10)

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_avg"],
                                              time=6,  # 6 aylık
                                              freq="W",  # T'nin frekans bilgisi.
                                              discount_rate=0.01)

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df["cltv"].sort_values(ascending=False).head(20)

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_df.groupby("cltv_segment")[["recency_cltv_weekly", "frequency", "monetary_cltv_avg"]].mean()


# GÖREV 5 BONUS: Tüm Süreci Fonksiyonlaştırınız

def create_cltv(dataframe):
    # Veriyi Hazırlama
    values = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
              "customer_value_total_ever_online"]
    for x in values:
        replace_with_threshold(df, x)
    # CLTV Veri Yapısının Oluşturulması
    df["total_number_purchase"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["total_number_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    date = df.columns[df.columns.str.contains("date")]
    df[date] = df[date].apply(pd.to_datetime)

    today_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["frequency"] = df["total_number_purchase"]
    cltv_df["monetary_cltv_avg"] = df["total_number_price"] / df["total_number_purchase"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    # BG/NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df["frequency"],
                                               cltv_df["recency_cltv_weekly"],
                                               cltv_df["T_weekly"])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df["frequency"],
                                               cltv_df["recency_cltv_weekly"],
                                               cltv_df["T_weekly"])
    # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"],
            cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                           cltv_df["monetary_cltv_avg"])
    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                                  cltv_df["frequency"],
                                                  cltv_df["recency_cltv_weekly"],
                                                  cltv_df["T_weekly"],
                                                  cltv_df["monetary_cltv_avg"],
                                                  time=6,  # 6 aylık
                                                  freq="W",  # T'nin frekans bilgisi.
                                                  discount_rate=0.01)
    # CLTV'ye Göre Segmentlerin Oluşturulması
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

df = df_.copy()
new_df = create_cltv(df)
new_df

