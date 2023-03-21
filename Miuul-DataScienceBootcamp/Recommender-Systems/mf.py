#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('Recommender-Systems/movie.csv')
rating = pd.read_csv('Recommender-Systems/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

# rate'lerin skalasını veriyoruz (1 ile 5 arasında)
reader = Reader(rating_scale=(1, 5))

# surprise'ın anlayaağı formata çeviriyoruz
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)


####################
# Adım 2: Modelleme
####################

# %75 test %25 train olarak ayırdık
trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)  # bu aşamada, p ve q ağırlıklarını bulduk
predictions = svd_model.test(testset)

# tahminde bulunduğumuz yapmamız beklenen ortalama hata
accuracy.rmse(predictions)  # RootMeanSquareError

# 1 numaralı id'nin BladeRunner filmine 4.0 vermiş biz 4.16 tahmin ettik
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]





