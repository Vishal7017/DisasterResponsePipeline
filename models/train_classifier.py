import sys
import pandas as pd
import re
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from joblib import dump
import pickle


def load_data(database_filepath):
    '''
    INPUT - database_filepath
    OUTPUT:
    X - features DataFrame
    y - responses DataFrame
    category_names - list of features available in feature DataFrame

    This function reads the table 'categorized_message' from a database db-file.
    This db-file is the output of the script 'process_data.py' and returns the
    X and y df and the list of features.
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('categorized_messages', engine)
    X = df['message']
    y = df.drop(columns=['id','message','original','genre', 'child_alone'])

    category_names = y.columns.tolist()

    return X, y, category_names

def tokenize(text):
    '''
    INPUT - text (string)
    OUTPUT - tokenized and cleansed text (string)

    This function tokenizes and cleanses a string by the following steps:
        1. any url will be replaced by the string 'urlplaceholder'
        2. any puctuation and capitalization will be removed
        3. the text will be tokenized
        4. any stopword will be removed
        5. each word will be first lemmatized and then stemmed
    '''

    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Remove punctuation characters
    text = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß ]', '', text.lower())

    # tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    # instantiate lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()

    # lemmatize and stemm
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT: None
    OUTPUT: GridSearchCV object

    This functions instantiates the model pipeline and performs a GridSearch
    Classifier: Gradient Boosting classifier
    GridSearch Parameters:
        learning_rate = [.05, .1]
        n_estimators = [50, 200]
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
    ])

    params_gbc = {'clf__estimator__n_estimators': [50, 200],
                  'clf__estimator__learning_rate': [.05, .1]
                 }

    cv = GridSearchCV(pipeline, param_grid=params_gbc, verbose=3, return_train_score=True)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT:
    model - trained model
    X_test - a np-array of testing features
    y_train - a np-array of testing responses
    category_names - a list of column names of the features df

    OUTPUT:
    a classification report for each feature will be printed out.
    '''

    #predicting y_test
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    y_test = pd.DataFrame(y_test, columns=category_names)

    print(model)
    print()
    print(classification_report(y_test.melt().value, y_pred.melt().value, zero_division=0))
    print('_______________________________________________________________________________________')
    print('Detailed classification report per feature')

    for col in category_names:
        print('{}\n{}'.format(col, classification_report(y_test[col], y_pred[col], zero_division=0)))

    print('____________________________________________________________________________________\n')

def save_model(model, model_filepath):
    # Save to file in the current working directory
    joblib_file = model_filepath
    dump(model, joblib_file, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train, X_test  = np.array(X_train), np.array(X_test)
        Y_train, Y_test = np.array(Y_train), np.array(Y_test)

        print('Building model...')
        model = build_model()

        print('Training model - this might take a while...')
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
