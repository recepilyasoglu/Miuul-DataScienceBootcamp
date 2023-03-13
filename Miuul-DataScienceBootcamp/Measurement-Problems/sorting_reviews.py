# SORTING REVIEWS

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)


# Up-Down Diff Score: (up ratings) - (down ratings)

# Review 1: 600 up, 400 down, total 1000
# Review 2: 5500 up, 4500 down, total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)


# Score = Average rating = (up ratings) / (all ratings)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up, 0 down, total 2
# Review 2: 100 up, 1 down, total 101

score_average_rating(2, 0)
score_average_rating(100, 1)  # patladı: sayı ve frekans yüksekliğini göz önünde bulunduramadı

