import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Schritt 1: Laden der Datensätze
train_df = pd.read_csv('../data/raw/dmml1_train.csv')
store = pd.read_csv('../data/raw/dmml1_stores.csv')

store['StoreType'] = store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
store['Assortment'] = store['Assortment'].map({'a': 0, 'c': 1, 'b': 2})
store['PromoInterval']= store['PromoInterval'].map({'Jan,Apr,Jul,Oct' : 0, 'Feb,May,Aug,Nov' : 1, 'Mar,Jun,Sept,Dec' : 2})
store["CompetitionDistance"].replace(np.nan,store["CompetitionDistance"].mean(),inplace=True)

train_df['StateHoliday'] = train_df['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3})
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df['Year'] = train_df['Date'].dt.year
train_df['Month'] = train_df['Date'].dt.month
train_df['Day'] = train_df['Date'].dt.day
train_df['WeekOfYear'] = train_df['Date'].dt.isocalendar().week
train_df['Weekend'] = np.where(train_df['DayOfWeek'].isin([6, 7]), 1, 0)  # Samstag = 6, Sonntag = 7
train_df['Quarter'] = train_df['Date'].dt.quarter
train_df['Season'] = train_df['Month'].apply(lambda month: (month%12 // 3 + 1))
# train_df['Season'].replace(to_replace=[1,2,3,4], value=['Winter', 'Frühling','Sommer','Herbst'], inplace=True)

train_df.drop('Date', axis=1, inplace=True)

train_df = pd.merge(train_df,store,how="inner",on="Store ID")
train_df.fillna(0,inplace=True)

X = train_df.drop(["Sales","Customers"], axis=1)
y = train_df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

# für StandardScaler
ss_colmnn = ['Store ID', 'Year', 'Month', 'Day', 'WeekOfYear', 'Promo2SinceWeek', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

# für One Hot Encoding
ohe_colmn = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']


SS = StandardScaler()
OHE = OneHotEncoder()
OE  = OrdinalEncoder()
LR = LinearRegression()


preprocess = make_column_transformer(
                                     (SS, ss_colmnn),
                                     (OHE, ohe_colmn),
                                     remainder='passthrough'
                                     )

RF = RandomForestRegressor(max_depth = None, n_jobs = -1, random_state = 42)

pipeline_rf = make_pipeline(preprocess,RF)

parameters = {'randomforestregressor__n_estimators': [200], 
              'randomforestregressor__max_depth': [None]}

# Rufen Sie Grid Search CV auf
Grid_Search_RFR = GridSearchCV(pipeline_rf, parameters)
Grid_Search_RFR.fit(X_train, y_train)

print(f"Accuracy Trainingsdaten: {Grid_Search_RFR.score(X_train, y_train)}")
print(f"Accuracy Testdaten: {Grid_Search_RFR.score(X_test, y_test)}")