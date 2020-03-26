
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
from plotly.graph_objects import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster', engine)

# load model
model = joblib.load("../models/multioutput.pkl")

#Plot figures
def return_figures():
    """Creates plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing plotly visualizations
    """
    #plot graph one
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one = [Pie(
    labels = genre_names,
    values = genre_counts,
    )]


    layout_one = dict(title = 'Distribution of Message Genres')
    
    #plot graph two

    cat_proportion = df[df.columns[4:]].mean().sort_values(ascending=False)[:10]
    cat_names = list(cat_proportion.index)
    
    graph_two = [Bar(
    x = cat_names,
    y = cat_proportion,
    )]

    layout_two = dict(title = 'Top 10 Categories of Disaster Response',
                yaxis = dict(title = "Proportion"))


    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures


# index webpage displays visuals from function return_figures() 
@app.route('/')
@app.route('/index')
def index():
    
    graphs = return_figures()
                    
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

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
