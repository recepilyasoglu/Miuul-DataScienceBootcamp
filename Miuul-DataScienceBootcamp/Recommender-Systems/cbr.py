#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması


#####################################
# 1. TF-IDF Matrisinin Oluşturulması
####################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_ = pd.read_csv("Recommender-Systems/movies_metadata.csv", low_memory=False)
df = df_.copy()
df.head()
df.shape

df["overview"].head()

# yaygın olmayan kelimeleri veri setinden çıkarmak istiyoruz, ölçüm değerleri taşımıyorlar
tfidf = TfidfVectorizer(stop_words="english")

df["overview"] = df["overview"].fillna("")

# fit, fit eder değişiklik yapar,
# transform ise bu değişikliği transform eder yani değerleri dönüştürür
tfidf_matrix = tfidf.fit_transform(df["overview"])
tfidf_matrix = tfidf_matrix.astype(np.float32)


tfidf_matrix.shape

df["title"].shape

tfidf.get_feature_names_out()

tfidf_matrix.toarray()


################################################
# 2. Cosine Similarity Matrisinin Oluşturulması
################################################

cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32),
                               tfidf_matrix.astype(np.float32))

cosine_sim.shape
cosine_sim[1]


##################################################################################
# 3. Benzerliklere Göre Önerilerin Yapılması(Recommendation Based on Similarities)
##################################################################################

indices = pd.Series(df.index, index=df["title"])

indices.index.value_counts()

# indexin içerisindeki title'larda duplicate olan var mı bunlara git
# sonra aynısını bir daha gördüğünde True der silmek için keep kısmında
# last diyerek sonuncuyu tut diyoruz, son çıkan filme ulaşmak için
indices = indices[~indices.index.duplicated(keep="last")]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

# Sherlock Holmes filmiyle elimizdeki bütün filmlerin benzerlik oranlarını çıkardık
similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

# filmlerin index bilgilerini aldık
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# yukarıdaki index bilgilerine sahip filmlerin bilgileri
df["title"].iloc[movie_indices]


###################################################################
# Çalışmanın Scriptinin Hazırlanması(Preparation of Working Script)
###################################################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri oluşturma
    indices = pd.Series(dataframe.index, index=dataframe[title])
    indices = indices[~indices.index.duplicated(keep="last")]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a göre benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe[title].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender("The Dark Knight Rises", cosine_sim, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words="english")
    dataframe["overview"] = dataframe["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    tfidf_matrix = tfidf_matrix.astype(np.float32)
    cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32),
                                   tfidf_matrix.astype(np.float32))
    return cosine_sim

