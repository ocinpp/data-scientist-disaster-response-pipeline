import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


application = Flask(__name__)
app = application

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def pretty_names(name_list):
    """
    Convert a list of string to a pretty display
    (replace '_' with space and change to title case)
    """
    return list(map(lambda x: x.replace('_', ' ').title(), name_list))

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = pretty_names(list(genre_counts.index))

    # sum of different categories
    cat_counts_all = df.sum(axis=0, numeric_only=True).sort_values(ascending=False).drop(labels=['id'])
    cat_names_all = pretty_names(cat_counts_all.axes[0].tolist())

    #print(cat_counts_all)

    cat_counts_asc = df.sum(axis=0, numeric_only=True).sort_values(ascending=True)
    cat_counts_asc = cat_counts_asc.drop(labels=['id'])
    cat_counts_asc = cat_counts_asc.head(10)
    cat_names_asc = pretty_names(cat_counts_asc.axes[0].tolist())

    cat_counts_desc = df.sum(axis=0, numeric_only=True).sort_values(ascending=False)
    cat_counts_desc = cat_counts_desc.drop(labels=['id'])
    cat_counts_desc = cat_counts_desc.head(10)
    cat_names_desc = pretty_names(cat_counts_desc.axes[0].tolist())

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    values=cat_counts_all,
                    labels=cat_names_all
                )
            ],

            'layout': {
                'title': 'Distribution of Categories'
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names_desc,
                    y=cat_counts_desc
                )
            ],

            'layout': {
                'title': 'Distribution of Categories (Top 10)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names_asc,
                    y=cat_counts_asc
                )
            ],

            'layout': {
                'title': 'Distribution of Categories (Last 10)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    classification_count = len(list(filter(lambda x: x == 1, classification_results.values())))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        classification_count=classification_count
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()