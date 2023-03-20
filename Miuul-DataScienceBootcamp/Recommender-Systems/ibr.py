###########################################
# Item-Based Collaborative Filtering
###########################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması


######################################
# Adım 1: Veri Setinin Hazırlanması
######################################

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv("Recommender-Systems/movie.csv")
rating = pd.read_csv("Recommender-Systems/rating.csv")
df = movie.merge(rating, how="left", on="movieId")
df.head()


##########################################
# Adım 2: User Movie Df'inin Oluşturulması
##########################################

df.shape
df["title"].nunique()
df["title"].value_counts().head()
comments_counts = pd.DataFrame(df["title"].value_counts())

# 1000 den az veya 1000'e eşit yorum sayısı alan filmler
rare_movies = comments_counts[comments_counts["title"] <= 1000].index

# rare'ın içendikelere bak onun dışındakiler getir, yani 1000 den fazla yorum alan filmler
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")




