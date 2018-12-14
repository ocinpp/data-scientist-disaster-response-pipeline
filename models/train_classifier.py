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
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    sents = sent_tokenize(text)
    tokens = []
    for sent in sents:
        # normalize case and remove punctuation
        text = re.sub(r"[^a-zA-Z0-9]", " ", sent.lower())
    
        # tokenize text
        work_tokens = word_tokenize(text)
    
        # lemmatize and remove stop words
        sent_tokens = [lemmatizer.lemmatize(word, pos='v') for word in work_tokens if word not in stop_words]
        
        # stemming
        sent_tokens = [PorterStemmer().stem(w) for w in sent_tokens]
        
        tokens = tokens + sent_tokens

    return tokens


def build_model():
    # try RandomForestClassifier
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
                    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)    

    for i, category_name in enumerate(category_names):
        print(category_name)
        print(classification_report(Y_test[category_name], y_pred[:,i]))


def save_model(model, model_filepath):
    # Open the file to save as pkl file
    my_pipeline_model_pkl = open(model_filepath, 'wb')
    
    # Dump the model with Pickle
    pickle.dump(model, my_pipeline_model_pkl)

    # Close the pickle instances
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