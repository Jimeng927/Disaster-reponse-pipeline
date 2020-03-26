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

The machine learning model evaluation results are show in the table below
```
                        precision    recall  f1-score   support

               related       0.84      0.95      0.89      3992
               request       0.81      0.50      0.62       913
                 offer       0.00      0.00      0.00        24
           aid_related       0.77      0.68      0.72      2161
          medical_help       0.76      0.08      0.15       403
      medical_products       0.80      0.08      0.14       251
     search_and_rescue       0.69      0.06      0.12       141
              security       0.00      0.00      0.00       104
              military       0.80      0.04      0.09       178
           child_alone       0.00      0.00      0.00         0
                 water       0.83      0.36      0.50       308
                  food       0.82      0.65      0.72       579
               shelter       0.83      0.37      0.51       452
              clothing       0.71      0.06      0.11        81
                 money       1.00      0.02      0.03       123
        missing_people       0.00      0.00      0.00        40
              refugees       0.50      0.01      0.02       176
                 death       0.90      0.19      0.32       224
             other_aid       0.62      0.03      0.06       710
infrastructure_related       0.00      0.00      0.00       348
             transport       0.76      0.11      0.19       245
             buildings       0.79      0.10      0.18       262
           electricity       1.00      0.05      0.09       110
                 tools       0.00      0.00      0.00        39
             hospitals       0.00      0.00      0.00        61
                 shops       0.00      0.00      0.00        28
           aid_centers       0.00      0.00      0.00        56
  other_infrastructure       0.00      0.00      0.00       235
       weather_related       0.87      0.70      0.77      1474
                floods       0.90      0.48      0.63       422
                 storm       0.80      0.51      0.62       506
                  fire       0.00      0.00      0.00        49
            earthquake       0.89      0.81      0.85       476
                  cold       0.78      0.06      0.11       115
         other_weather       0.86      0.02      0.04       272
         direct_report       0.76      0.36      0.49      1045

             micro avg       0.82      0.53      0.65     16603
             macro avg       0.56      0.20      0.25     16603
          weighted avg       0.76      0.53      0.57     16603
           samples avg       0.67      0.48      0.51     16603
```

<a name="ac"></a>
## 5. Licensing, Authors, and Acknowledgements
Licensing: [MIT](https://choosealicense.com/licenses/mit/)
Acknowledgements: The dataset is from [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/). This project is adapted from Udacity's Data Science Nanodegree. 
