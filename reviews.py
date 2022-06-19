# Import packages
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np

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
    csv_files = glob.glob(os.path.join(cur_work_dir + folder, "*.csv"))
    combined = list()
    for name in csv_files:
        df = pd.read_csv(name)
        df['city'] = name.split("\\")[-1][:-4]
        combined.append(df)
    concatenated = pd.concat(combined)
    concatenated.dropna(subset=['comments'], inplace=True)

    return concatenated


def detect_lang(df):
    """
    INPUT:
        df: pandas DataFrame - dataframe a column of which will be used
        for language detection

    OUTPUT:
        df: pandas DataFrame - dataframe with a new column with detected languages

    Description:
        This function takes a dataframe and append it with column with detected
        languages

    """

    df['language'] = (df['comments'].apply(langid.classify)).apply(lambda x: x[0])

    return df

def sentiment_scores_eng(df, language_col):
    """
    INPUT:
        df: pandas DataFrame - dataframe with a detected languages column
        language_col: pandas Series - column with assigned detected languages

    OUTPUT:
        df_english: pandas DataFrame - dataframe with polarity score and assigned
        classification of comments either positive (1) or negative (0)

    Description:
        This function takes a dataframe and append it with polarity score and
        classification of comments

    """

    df_english = df[df[language_col]=="en"]

    analyser = SentimentIntensityAnalyzer()

    df_english['compound_score'] = df_english.comments\
    .apply(lambda x: analyser.polarity_scores(x)['compound'])

    df_english['comp_score'] = df_english['compound_score']\
    .apply(lambda c: 1 if c >=0 else 0)

    return df_english

def store_file_csv(df, file_name_csv):
    """
    INPUT:
        df: pandas DataFrame - clean dataframe
        file_name_csv: str - name of output file

    OUTPUT:
        None

    Description:
        Saves file to csv
    """
    # Safe resulting dataframe into csv file
    df.to_csv(file_name_csv + '.csv',index= False)

def main():
    if len(sys.argv) == 3:

        folder, file_name_csv = sys.argv[1:]

        print('Loading data...')
        df = read_file(folder)
        print('The dataframe has {} rows and {} columns'.format(df.shape[0], df.shape[1]))

        print('Detecting languages...')
        df = detect_lang(df)
        print((df.language.value_counts()/df.shape[0])[:5])

        print('Calculating sentiment scores...')
        df = sentiment_scores_eng(df, language_col= 'language')
        print(df.comp_score.value_counts())

        print('Saving to csv...')
        store_file_csv(df, file_name_csv)

        print('Saved to csv!!!')

    else:
        print('Please provide folder: string - name of the folder located in the working directory, '\
        'for example "\Reviews" and file_name_csv')

if __name__ == '__main__':
    main()
