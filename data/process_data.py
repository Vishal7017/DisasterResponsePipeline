import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath containing the message information
    categories_filepath - filepath containing the categorization of messages

    OUTPUT:
    df - joined messages and categories information

    This function will load the messages and categories information from a csv
    file and join them together.
    '''
    #loading messages and categories based on input paths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #joining messages and categories based on id
    df = pd.merge(messages, categories, how='inner', on='id')

    return df


def clean_data(df):
    '''
    INPUT:
    df - joined messages and categories information

    OUTPUT:
    df - joined messages and categories information

    This function will load the messages and categories information from a csv
    file and join them together.
    '''

    #splitting the category column into separate columns per category
    categories = df['categories'].str.split(';', expand=True)

    #REFACTOR
    #generating a dictionary per column number to a category
    list_first_row = np.array(categories.head(1))[0].tolist()

    column_mapping = {}
    for i in range(len(list_first_row)):
        column_mapping[i] = re.sub('-.', '', list_first_row[i])

    categories.rename(columns=column_mapping, inplace=True)

    #removing category name per data entry and keeping only the number
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = np.sign(categories[column].astype(int))

    #dropping initial categories column and joining processed category df
    df = pd.concat([df.drop(columns='categories'), categories], axis=1)

    #dropping duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - data frame containing processed categorized messages
    database_filename - filename as .db file where to store the information

    This function will load the messages and categories information from a csv
    file and join them together.
    '''

    #creating sql engine to save df in sql db
    engine = create_engine('sqlite:///'+database_filename)
    #saving df in sql db
    df.to_sql('categorized_messages', engine, if_exists='replace', index=False)


def main():
    '''
    Performing the whole data processing:
    1. Loading and joining files based on input parameters:
        First argument = messages file
        Second argument =  categorization file
    2. Data preprocessing: creating dummy variables based on category input
    3. Storing df in database based on input parameter: third argument = database name

    if the number of arguments does not match the expected numer of arguments (3)
    an exception will be thrown.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
