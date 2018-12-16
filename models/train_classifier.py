import sys
import pandas as pd
import nltk
import re
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """
    Load data from database_filepath and
    return X, Y and category_names
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize the input text by normalizing, removing punctuations,
    lemmatizing and removing stop words
    """
    sents = sent_tokenize(text)
    tokens = []
    for sent in sents:
        # normalize case and remove punctuation
        text = re.sub(r"[^a-zA-Z0-9]", " ", sent.lower())

        # tokenize text
        word_tokens = word_tokenize(text)

        # lemmatize and remove stop words
        sent_tokens = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens if word not in stop_words]

        # stemming
        sent_tokens = [PorterStemmer().stem(w) for w in sent_tokens]

        tokens = tokens + sent_tokens

    return tokens


def build_model():
    """
    Build a machine learning pipeline using
    CountVectorizer, TfidfTransformer, MultiOutputClassifier with RandomForestClassifier
    """
    # try RandomForestClassifier
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,1))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
                    ])
    return pipeline


def find_best_model(model, X_train, y_train):
    """
    Use GridSearchCV to find the best parameters for the model
    """

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [10, 100]
    }

    cv = GridSearchCV(model, param_grid=parameters)
    cv.fit(X_train, y_train)

    print('best paramters:')
    print(cv.best_params_)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model using the given X_test and Y_test
    """

    y_pred = model.predict(X_test)

    for i, category_name in enumerate(category_names):
        # print the category name, f1 score, precision and recall
        print(category_name)
        print(classification_report(Y_test[category_name], y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Save the model as a pkl file specified by model_filepath
    """

    # open the file to save as pkl file
    my_pipeline_model_pkl = open(model_filepath, 'wb')

    # dump the model with Pickle
    pickle.dump(model, my_pipeline_model_pkl)

    # close the pickle instances
    my_pipeline_model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('GridCVSearch...')
        model = find_best_model(model, X_train, Y_train)

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