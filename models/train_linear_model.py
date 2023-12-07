# Importieren der erforderlichen Bibliotheken
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from pandas.api.types import CategoricalDtype

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(train_data_df, store_data_df):
    median_distance = store_data_df['CompetitionDistance'].median()
    store_data_df['CompetitionDistance'].fillna(median_distance, inplace=True)
    merged_data = train_data_df.merge(store_data_df, on='Store ID', how='left')
    merged_data.drop(columns=['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], inplace=True)


    return merged_data

def feature_engineering(merged_data):
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    merged_data['Year'] = merged_data['Date'].dt.year
    merged_data['Month'] = merged_data['Date'].dt.month
    merged_data['Day'] = merged_data['Date'].dt.day
    merged_data['WeekOfYear'] = merged_data['Date'].dt.isocalendar().week
    merged_data['Weekend'] = np.where(merged_data['DayOfWeek'].isin([6, 7]), 1, 0)

    cat_type = CategoricalDtype(categories=['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'], ordered=True)
    merged_data['Weekday'] = merged_data['Date'].dt.day_name(locale='de_DE').astype(cat_type)
    merged_data['Quarter'] = merged_data['Date'].dt.quarter
    merged_data['DayOfYear'] = merged_data['Date'].dt.dayofyear
    merged_data['DayOfMonth'] = merged_data['Date'].dt.day
    merged_data['Season'] = merged_data['Month'].apply(lambda month: (month%12 // 3 + 1))
    merged_data['Season'].replace(to_replace=[1,2,3,4], value=['Winter', 'Frühling','Sommer','Herbst'], inplace=True)

    merged_data.drop('Date', axis=1, inplace=True)

    return merged_data


def one_hot_encode_and_scale(merged_data):
    categorical_columns = ['StateHoliday', 'StoreType', 'Assortment', 'Season', 'Weekday']
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_columns = encoder.fit_transform(merged_data[categorical_columns])
    encoded_columns_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))
    merged_data_encoded = pd.concat([merged_data, encoded_columns_df], axis=1)
    merged_data_encoded.drop(categorical_columns, axis=1, inplace=True)

    numerical_columns = [col for col in merged_data_encoded.columns if col not in ['Store ID', 'Open', 'Promo', 'SchoolHoliday', 'Promo2', 'Weekend'] and merged_data_encoded[col].nunique() > 2]
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(merged_data_encoded[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)
    for col in numerical_columns: 
        merged_data_encoded[col] = scaled_numerical_df[col]

    return merged_data_encoded

def optimize_features(X_train, y_train):
    estimator = LinearRegression()
    selector = RFE(estimator)
    n_features = X_train.shape[1]
    param_grid = {'n_features_to_select': list(range(1, n_features + 1, 1))}
    grid_search = GridSearchCV(selector, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def select_features(X_train, y_train, grid_search):
    best_n_features = grid_search.best_params_['n_features_to_select']
    selector = RFE(LinearRegression(), n_features_to_select=best_n_features)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_]
    dropped_features = X_train.columns[~selector.support_]
    X_train_selected = X_train.loc[:, selector.support_]

    return X_train_selected, selected_features, dropped_features

def train_final_model(X_train_selected, y_train):
    final_model = LinearRegression()
    final_model.fit(X_train_selected, y_train)
    return final_model


def evaluate_model(model, X_test, y_test, selected_features):
    X_test_selected = X_test[selected_features]

    y_pred = model.predict(X_test_selected)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r_squared = r2_score(y_test, y_pred)

    return rmse, r_squared



def main():
    train_data_df = load_data('../data/raw/dmml1_train.csv')
    store_data_df = load_data('../data/raw/dmml1_stores.csv')
    merged_data = preprocess_data(train_data_df, store_data_df)
    merged_data = feature_engineering(merged_data)
    merged_data_encoded = one_hot_encode_and_scale(merged_data)

    X = merged_data_encoded.drop(['Sales', 'Customers'], axis=1)
    y = merged_data_encoded['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = optimize_features(X_train, y_train)
    X_train_selected, selected_features, dropped_features = select_features(X_train, y_train, grid_search)

    final_model = train_final_model(X_train_selected, y_train)

    rmse, r_squared = evaluate_model(final_model, X_test, y_test, selected_features)


    print(f"Optimale Anzahl von Features: {len(selected_features)} von {X_train.shape[1]}")
    print(f"Ausgewählte Features: {selected_features.tolist()}")
    print(f"Weggelassene Features: {dropped_features.tolist()}")
    print(f"Root Mean Squared Error (RMSE) des finalen Modells: {rmse}")
    print(f"R-Quadrat (R²) des finalen Modells: {r_squared}")

if __name__ == "__main__":
    main()
