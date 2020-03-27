import pandas as pd
from plotly.graph_objects import Bar
from plotly.graph_objects import Pie
from sqlalchemy import create_engine

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster', engine)

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
