############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması


#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

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

# veri setinden rastgele user seçmek
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#################################################################################
# Adım 2: İzlenen Filmleri Getirme Uygulaması(Practical to Bring Watched Movies)
#################################################################################
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user]

# user'in puan verdiği filmleri alalım, NAN olmayan
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# satırlarında (index) user'i bul, sütunda da yazılı filmi bul (sağlama yapmak için)
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Schindler's List (1993)"]

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Sabrina (1995)"]

len(movies_watched)  # toplam izlediği film sayısı


######################################################################################
# Adım 3: Aynı Filmi İzleyen Diğer Kullanıcılar(Other Users Watching the Same Movies)
######################################################################################

# artık sadece izlenen filmleri aldık
movies_watched_df = user_movie_df[movies_watched]

# her bir kullanıcın kaç tane film izlediği bilgisi
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

# user'imiz ile en az 20 ortak filmi izleyen kullanıcılar
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# hepsini izleyen kullanıcı
user_movie_count[user_movie_count["movie_count"] == 33].count()

# en az 20 film izleyen kullanıcıların id bilgileri
user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


##########################################################################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
##########################################################################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. User ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız







