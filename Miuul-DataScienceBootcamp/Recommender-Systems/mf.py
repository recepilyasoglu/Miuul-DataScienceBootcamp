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


#######################
# Adım 3: Model Tuning
#######################

param_grid = {'n_epochs': [5, 10, 20],  # iterasyon sayısı, kaç defa ağırlıkları güncellicem
              'lr_all': [0.002, 0.005, 0.007]}  # bütün parametreler için öğrenme oranı

# param_gird= yukarıdaki iki hiper parametrenin olası tüm kombinasyonlarını dene
# measures= gerçek değerler ile tahmin edilen değerler arasındaki farklarının karelerinin ortalamasını
# veya karekökünü al
# cv= 3 katlı çapraz doğrulama yap, veri setini 3'e böl, 2 parçasıyla model kur, 1 parçasıyla test et
# sonra diğer iki parçasıyla model urup dışarda bıraktığınla test et, aynısını sonuncuya da yap
# n_jobs= işlemcileri full performasn ile kullan
# joblib_verbose= işlemler gerçekleşirken bana raporlama yap
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse']  # en iyi skorumuz
gs.best_params['rmse']  # en iyi parametreler


################################
# Adım 4: Final Model ve Tahmin
################################

# model oluşturma basamağına tekrar gidip daha iyi parametreler bulduk dememiz lazım aslında
dir(svd_model)
svd_model.n_epochs

# iki yıldız ve sözlük yapısını gönderdiğimizde, modeli yenideğerler ile oluşturur
svd_model = SVD(**gs.best_params['rmse'])

# bütün veri setini train set'e çevirmiş olduk, çünkü öğreneceğimizi öğrendik zaten
data = data.build_full_trainset()
svd_model.fit(data)  # modeli fit ettik

# Blade Runner için tahmin 4.20, normalde puan 4.0 - çok iyi değil, çok kötü de değil
svd_model.predict(uid=1.0, iid=541, verbose=True)


