import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads message and categories dataset into pandas DataFrames,
    and combine them into a new dataframe

    INPUT:
    messages_filepath: filepath of 'messages.csv'
    categories_filepath: filepath of 'categories.csv'

    OUTPUT:
    df: A merged dataset using the common id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    This function split [categories] column into separate category columns,
    convert category values to just numbers 0 or 1,
    and replace [categories] column in df with new category columns

    INPUT:
    df: the merged dataframe

    OUTPUT:
    df: a cleaned dataframe that is ready for Machine Learning Pipeline
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # drop rows with values equal to 2
    df = df[(df != 2).all(axis=1)]

    return df


def save_data(df, database_filename):
    '''
    This function saves the clean dataset into an sqlite database

    INPUT:
    df: the cleaned database
    database_filename: filename.db that contains the clean dataset

    OUTPUT:
    None

    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster', engine, index=False)



def main():
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
