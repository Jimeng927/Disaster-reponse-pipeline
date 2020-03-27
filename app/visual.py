import pandas as pd
from plotly.graph_objects import Bar
from plotly.graph_objects import Pie
from collections import Counter
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster', engine)

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

    message = " ".join(df["message"])
    cleaned_message = tokenize(message)
    word_count_list = Counter(cleaned_message).most_common(10)
    words = list((dict(word_count_list)).keys())
    count = list((dict(word_count_list)).values())
    
    graph_two = [Bar(
    x = words,
    y = count,
    )]

    layout_two = dict(title = 'Top 10 Most common words in messages',
                yaxis = dict(title = "counts"))

    #plot graph three

    cat_proportion = df[df.columns[4:]].mean().sort_values(ascending=False)
    cat_names = list(cat_proportion.index)
    
    graph_three = [Bar(
    x = cat_names,
    y = cat_proportion,
    )]

    layout_three = dict(title = 'Categorie Distribution of Disaster Response',
                yaxis = dict(title = "Proportion"))


    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))

    return figures
