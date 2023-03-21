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
pd.set_option('display.max_columns', 500)
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


##############################################################################
# Adım 3: Item-Based Film Önerilerinin Yapılması(Making Movie Recommendations)
##############################################################################

movie_name = "Matrix, The (1999)"
movie_name2 = "Ocean's Twelve (2004)"

movie_name = user_movie_df[movie_name]
movie_name2 = user_movie_df[movie_name2]

user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
user_movie_df.corrwith(movie_name2).sort_values(ascending=False).head(10)

# Rastgele film almak
movie_name3 = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name3 = user_movie_df[movie_name3]
user_movie_df.corrwith(movie_name3).sort_values(ascending=False).head(10)

# Keyword'e göre ilgili dataframe'in içindeki filmler de arama
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Sherlock", user_movie_df)
check_film("Insomnia", user_movie_df)


########################################################################
# Adım 4: Çalışma Scriptinin Hazırlanması(Preparation of Working Script)
########################################################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Recommender-Systems/movie.csv")
    rating = pd.read_csv("Recommender-Systems/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)


movie_name3 = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name3, user_movie_df)

