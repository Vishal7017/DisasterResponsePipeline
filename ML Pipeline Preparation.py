#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


# In[3]:


# load data from database
engine = create_engine('sqlite:///myProjectDatabase.db')
df = pd.read_sql_table('Categories_Table', engine)  

X = df['message']
y = df[df.columns.difference(['id', 'message','original', 'genre'])]


# ### 2. Write a tokenization function to process your text data

# In[5]:


def tokenize(text):
    text = re.sub(r"[^a-z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens

print(tokenize('Weather update - a cold front from Cuba that could pass over Haiti'))


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[ ]:


#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train pipeline
fitted_pipeline = pipeline.fit(X_train, y_train)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:


# predict on test data
y_pred_np = fitted_pipeline.predict(X_test)


# In[ ]:


#converting to dataframe
y_pred = pd.DataFrame (y_pred_np, columns = y_test.columns)
display(y_test.head())
display(y_pred.head())


# In[ ]:


for col in y_test.columns:
    #returning dictionary from classification report
    #print(col)
    #display(y_test[col].head())
    print("Category:", col)
    class_col= classification_report(y_true = y_test.loc [:,col], y_pred = y_pred.loc [:,col])
    print(class_col)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


# specify parameters for grid search
parameters = {
        'vect__analyzer': ['word'],
        'vect__max_features': [5, 50],
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 5]
    }

# create grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


# predict on test data
y_pred_grid = cv.predict(X_test)

#converting to dataframe
y_pred_grid_df = pd.DataFrame (y_pred_grid, columns = y_test.columns)
display(y_test.head())
display(y_pred_grid_df.head())


# In[ ]:


for col in y_test.columns:
    #returning dictionary from classification report
    #print(col)
    #display(y_test[col].head())
    class_col= classification_report(y_true = y_test.loc [:,col], y_pred = y_pred_grid_df.loc [:,col])
    print(class_col)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:



filename = 'finalized_model.sav'
pickle.dump(cv, open(filename, 'wb'))
 


# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




