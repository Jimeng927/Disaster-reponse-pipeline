# Disaster Response Pipeline Project

1. [Project Overview.](#proj)
2. [Installation.](#inst)
3. [File Descriptions.](#file)
4. [Results.](#res)
5. [Licensing, Authors, and Acknowledgements.](#ac)

<a name="proj"></a>
## 1. Project Overview

This project analyzes large dataset from past disasters and uses machine learning to plan for disaster response. The results are delivered via a Flask web app. 

The project includes three components:

1. **ETL Pipeline:** The first part of the data pipeline is the ETL(Extract, Transform, and Load) process. Here, the dataset is read, cleaned, and then stored in a SQLite database. 

2. **Machine Learning Pipeline:** The machine learning pipeline uses NLTK, scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). And then the model is exported to a pickle file.

3. **Flask App**: The results are visulized and displayed in a Flask web app.


<a name="inst"></a>
## 2. Installation

Python 3.7.2 

```python
import sys
from sqlalchemy import create_engine
import pandas 
import numpy
import nltk
import re
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
import json
import plotly
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
from plotly.graph_objects import Pie
from sklearn.externals import joblib
```

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="file"></a>
## 3. File Description
```bash
├── app         
│   ├── templates                         
│   │   ├── master.html                  # master html page 
│   │   ├── go.html                      # master/go html page that displays the search bar result
│   └── run.py                           # Flask web app
├── data
│   ├── disaster_messages.csv            # Dataset containing original messages
│   ├── disaster_categories.csv          # Dataset encoded with 36 different categories related to disaster response
│   ├── process.py                       # ETL pipeline
│   └── DisasterResponse.db              # cleaned dataset stored in SQLite database 
└── models
    └── train_classifier.py              # Machine learning pipeline
```

<a name="res"></a>
## 4. Results

<a name="ac"></a>
## 5. Licensing, Authors, and Acknowledgements
Licensing: [MIT](https://choosealicense.com/licenses/mit/)
Acknowledgements: The dataset is from [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). This project is adapted from Udacity's Data Science Nanodegree. 
