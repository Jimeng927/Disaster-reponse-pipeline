import sys
import nltk
nltk.download('punkt')
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    This function loads dataset from database,
    and define feature and target variables X and y

    INPUT:
    database_filepath:
    OUTPUT:
    X: [message] column as input for machine learning
    y: 36 categories as output classification results
    category_names: name for each classification output
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Disaster', con=engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    This function processes the text data with tokenization, lemmatization,
    and stopwords removal

    INPUT:
    text: the original text to be cleaned
    OUTPUT:
    text: the normalized, tokenized, and lemmatized word list from text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens


def build_model():
    '''
    Build model with GridSearchCV

    OUTPUT:
    Tuned model after using GridSearchCV
    '''
    # build a machine learning pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    # hyper parameter grid

    parameters = {
        'clf__estimator__max_depth' : [2,4,6],
        'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        'clf__estimator__n_estimators': [50, 100, 200]
             }

    # create model
    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, n_jobs = 4)

    model = pipeline
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model: trained ML models
    X_test: test features
    y_test: test output
    category_name: name for each classification output
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
