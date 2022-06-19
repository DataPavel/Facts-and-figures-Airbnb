# Import packages
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import langid
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
import glob
import sys



def read_file(folder):
    """
    INPUT:
        folder: string - name of the folder located in the working directory,
        for example '\\Reviews'

    OUTPUT:
        concatenated: pandas DataFrame - dataframe with all read in files concatenated

    Description:
        This function takes files from the specified directory and read them
        into a dataframe

    """

    cur_work_dir = os.getcwd()
    csv_files = glob.glob(os.path.join(cur_work_dir + '\\' + folder, "*.csv"))
    combined = list()
    for name in csv_files:
        df = pd.read_csv(name, low_memory=False)
        df['city'] = name.split("\\")[-1][:-4]
        combined.append(df)
    concatenated = pd.concat(combined)

    return concatenated


def clean_data_calendar(df):
    """
    INPUT:
        df: pandas DataFrame - calendar dataframe

    OUTPUT:
        clean_df: pandas DataFrame - clean dataframe

    Description:
        This function takes a calendar dataframe and convert each column into
        correct data type

    """

    df['price'] = df['price'].str.replace('$', '', regex = False).str.replace(',', '').astype(float)

    df['adjusted_price'] = df['adjusted_price'].str.replace('$', '', regex = False).str.replace(',', '').astype(float)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['available'] = np.where(df['available'] == 't', True, False)
    df['month'] = df['date'].dt.to_period('M')
    df.dropna(subset=['adjusted_price'], inplace=True)

    clean_df = df
    return clean_df




def clean_listings(df):
    """
    INPUT:
        df: pandas DataFrame - listings dataframe

    OUTPUT:
        df: pandas DataFrame - clean listings dataframe

    Description:
        This function takes the listings dataframe and clean it, please
        follow the comments inside the code

    """

    # Remove columns
    to_be_removed = ['listing_url', 'scrape_id', 'name', 'description',
    'picture_url', 'host_url', 'host_name', 'host_thumbnail_url',
    'host_picture_url', 'calendar_last_scraped', 'license']
    df.drop(to_be_removed, axis=1, inplace = True)

    # Change to datetime
    to_date = ['last_scraped', 'host_since', 'first_review', 'last_review']
    for i in to_date:
        df[i] = pd.to_datetime(df[i], format='%Y-%m-%d')

    # From datetime to integers year month and display
    date_vars = df.select_dtypes(include=['datetime64'])
    for i in list(date_vars):
        df[i+'_day'] = df[i].dt.day
        df[i+'_month'] = df[i].dt.month
        df[i+'_year'] = df[i].dt.year
    df.drop(date_vars, axis = 1, inplace = True)

    # Changes to float remove % and divide by 100
    to_float_remove_per = ['host_response_rate', 'host_acceptance_rate']
    for i in to_float_remove_per:
        df[i] = df[i].str.replace('%', '').astype(float)/100

    # Changes to float remove $ and delete commas
    to_float_remove_usd = ['price']
    df['price'] = df['price'].str.replace('$', '', regex = False)\
    .str.replace(',', '').astype(float)

    # Changes to boolean
    bool_ifany = ['host_about', 'neighborhood_overview']
    to_bool = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
           'has_availability', 'instant_bookable']
    for i in to_bool:
        df[i] = np.where(df[i] == 't', 1, 0)
    for i in bool_ifany:
        df[i] = np.where(df[i] != np.nan, 1, 0)

    # Split host location
    df.host_location = df.host_location.str.split(', ', expand = True)[0]

    # Split neighborhood
    df.neighbourhood = df.neighbourhood.str.split(', ', expand = True)[0]

    # Split bathroom text

    bath = df.bathrooms_text.str.split(' ', expand = True)\
    .rename(columns = {0: 'one', 1: 'two', 2: 'three'}).reset_index().drop('index', axis =1)

    bath['nr_baths'] = bath['one']\
    .apply(lambda x: 0.5 if x in ['Private', 'Half-bath', 'Shared']  else x).astype(float)

    bath['type_baths'] = bath['two']\
    .apply(lambda x: 'shared' if x == 'half-bath'  else (np.nan if x in ['baths', 'bath', None] else x ))

    bath.drop(['one', 'two', 'three'], axis=1, inplace=True)
    bath = bath.reset_index(drop=True)


    # Delete split columns

    deleted = ['host_verifications', 'bathrooms', 'bathrooms_text',
    'amenities', 'calendar_updated', 'host_id']

    df = df.drop(deleted, axis = 1)
    df = df.reset_index(drop = True)

    # Concat with bath
    df = pd.concat([df, bath], axis = 1,)

    # Rename columns
    df.rename(columns = {'id':'listing_id'}, inplace = True)
    df = df[df['price']!=0]

    df = remove_outliers_group(df, 'city', 'price')

    return df


def remove_outliers_group(df, group_column, num_column):
    """
    INPUT:
        df: pandas DataFrame - dataframe to remove outliers from
        group_column: str - name of a column for groupping
        num_column: str - name of a column from which outliers should be removed

    OUTPUT:
        new_df: pandas DataFrame - dataframe with removed outlisers

    Description:
        This function removes outliers from specified column based on groupping
        of another column

    """
    col_list = df[group_column].unique()
    dfs = []
    for i in col_list:
        num_col = df[df[group_column]== i][num_column]
        upper_limit = np.quantile(num_col,0.75) + (np.quantile(num_col,0.75)\
         - np.quantile(num_col,0.25))*1.5
        new_df = df[(df[group_column]== i) & (df[num_column] <= upper_limit)]
        dfs.append(new_df)
    new_df = pd.concat(dfs, axis=0)
    return new_df


def numeric_nan_with_mean(df):
    """
    INPUT:
        df: pandas DataFrame - dataframe to replace nan with mean

    OUTPUT:
        df: pandas DataFrame - dataframe with repleced nan with mean

    Description:
        This function replaces nan with mean values

    """
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in list(num_vars):
        df[col] = df.loc[:][col].fillna((df[col].mean()))

    return df

def encode_strings(df):
    """
    INPUT:
        df: pandas DataFrame - dataframe to encode strings and bools

    OUTPUT:
        df: pandas DataFrame - dataframe with encoded strings and bools

    Description:
        This function takes strings and bools and encodes them with 1 and 0
    """
    cat_vars = df.select_dtypes(include=['object', 'bool']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var,\
        prefix_sep='_', drop_first=True)], axis=1)
    return df

def prepare_x_y_for_modeling(df):
    """
    INPUT:
        df: pandas DataFrame - clean df for modeling

    OUTPUT:
        X: pandas DataFrame - dataframe with features
        y: pandas Series - pandas series with target variable

    Description:
        This function divides a detaframe by 2 dataframes: fetatures df and target df
    """
    X = df.drop(['price', 'listing_id'], axis=1)
    y = df['price']
    return X, y

def modeling(X, y):
    """
    INPUT:
        X: pandas DataFrame - dataframe with features
        y: pandas Series - pandas series with target variable

    OUTPUT:
        pipe: trainied lasso model
        X_train: pandas DataFrame - X_train dataframe

    Description:
        This function divides X and y by train and test set, create ml pipeline,
        train the model, predicts and calculates r quared score
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    pipe = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(fit_intercept=True))])
    pipe.fit(X_train, y_train)
    y_test_preds = pipe.predict(X_test)
    y_train_preds = pipe.predict(X_train)
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)
    print('Test R2 score is {:,.2f}'.format(test_score), 'Train R2 score is {:,.2f}'.format(train_score))
    return X_train, pipe

def coef_weights(pipe, X_train):
    """
    INPUT:
    coefficients - the coefficients of the linear model
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)

    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the
    variable attached to the coefficient.
    """

    coefs_df = pd.DataFrame()
    coefs_df['variable'] = X_train.columns
    coefs_df['coefficient'] = np.abs(pipe.named_steps['lasso'].coef_)
    coefs_df = coefs_df.sort_values('coefficient', ascending=False)[['variable', 'coefficient']]
    return coefs_df

def city_price_coefs(df, city):
    """
    INPUT:
        df: pandas DataFrame - dataframe
        city: str - for populate a column in resulting dataframe

    OUTPUT:
        importance: pandas dataframe - dataframe with city, features, coefficients columns

    Description:
        This function creates a dataframe that displays coefficients for each feature variable
    """
    X, y = prepare_x_y_for_modeling(df)
    X = numeric_nan_with_mean(X)
    X = encode_strings(X)
    X_train, pipe = modeling(X,y)
    importance = coef_weights(pipe, X_train)
    importance['city'] = city
    return importance

def create_table_sentiments(df):
    """
    INPUT:
        df: pandas DataFrame - dataframe with sentiment score

    OUTPUT:
        list_of_tuple: list of tuples - column names of multiindex column
        reviews_num: pandas DataFrame - resulting dataframe

    Description:
        This function creates a dataframe that displays proportions of negative reviews for neighborhood
    """
    neg_count = df.query('avg_score<0').groupby(['city', 'neighbourhood_cleansed'])['avg_score'].count().reset_index(name='neg_count').sort_values(by=['city', 'neg_count'], ascending=False)
    neg_count_area = neg_count.neighbourhood_cleansed.unique()
    all_count = df[df['neighbourhood_cleansed'].isin(neg_count_area)].groupby(['city', 'neighbourhood_cleansed'])['avg_score'].count().reset_index(name='all_count').sort_values(by=['city', 'all_count'], ascending=False)

    review_prct = neg_count.merge(all_count, how='left', on=['city', 'neighbourhood_cleansed'])
    review_prct['%'] = review_prct['neg_count'] / review_prct['all_count']*100
    review_prct = review_prct.sort_values(by=['city', '%'], ascending=False)
    review_prct.rename(columns = {'neighbourhood_cleansed':'Neighbourhood'}, inplace = True)
    cities_list = review_prct.city.unique()
    for_table = review_prct[['Neighbourhood', '%']]

    list_of_tuple = []
    num_city = list()
    for city in cities_list:
        list_of_tuple.append((city, for_table.columns[0]))
        list_of_tuple.append((city, for_table.columns[1]))
        city = review_prct[review_prct['city']==city][['Neighbourhood', '%']][:5].reset_index(drop=True)
        num_city.append(city)

    reviews_num = pd.concat(num_city, axis=1)
    cols = pd.MultiIndex.from_tuples(list_of_tuple)
    reviews_num.columns = cols
    return list_of_tuple, reviews_num

def create_table_weights(df, n_rows):
    """
    INPUT:
        df: pandas DataFrame - clean dataframe

    OUTPUT:
        list_of_tuple: list of tuples - column names of multiindex column
        coefficients: pandas DataFrame - resulting dataframe

    Description:
        This function creates a dataframe that displays coefficients for feature variables by each city
    """
    for_table2 = ['variable', 'coefficient']
    cities = df.city.unique()

    list_of_tuple = list()
    num_city = list()
    print('Lasso Regression...')
    for i in cities:
        print('Calculating score for {}'.format(i))
        city_coef = city_price_coefs(df[df['city']==i], i).reset_index(drop=True)[['variable', 'coefficient']][:n_rows]
        list_of_tuple.append((i, for_table2[0]))
        list_of_tuple.append((i, for_table2[1]))
        num_city.append(city_coef)

    coefficients = pd.concat(num_city, axis=1)
    cols = pd.MultiIndex.from_tuples(list_of_tuple)
    coefficients.columns = cols
    return list_of_tuple, coefficients
