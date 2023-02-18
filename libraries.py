# Importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

# Reading train and test data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

X=train.drop(columns=['target'])
y=train['target']

# Splitting the train and test data.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


X_train=X_train.fillna('empty')
X_test=X_test.fillna('empty')

vectorizer_keyword = CountVectorizer(max_features=200)
vectorizer_location = CountVectorizer(max_features=200)


X_train_location = vectorizer_location.fit_transform(X_train['location'])
X_test_location  = vectorizer_location.transform(X_test['location'])

X_train_keyword = vectorizer_keyword.fit_transform(X_train['keyword'])
X_test_keyword = vectorizer_keyword.fit_transform(X_test['keyword'])

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text'])
X_test_tfidf= tfidf_vectorizer.transform(X_test['text'])
X_train_total=hstack([X_train_location,X_train_keyword,X_train_tfidf]).toarray()
X_test_total=hstack([X_test_location,X_test_keyword,X_test_tfidf]).toarray()

train_data=xgb.DMatrix(data=X_train_total,label=y_train)
validation_data=xgb.DMatrix(data=X_test_total,label=y_test)


param={
    'learning_rate':0.2,  
    'max_depth':12,
    'min_child_weight':5,
    'gamma':0.4
    
}

bst = xgb.train(param, train_data)


##Inference code
y_pred=bst.predict(validation_data)

y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

print(accuracy_score(y_test,y_pred))

joblib.dump(vectorizer_keyword,'vectorizer_keyword.pkl')

joblib.dump(vectorizer_location,'vectorizer_location.pkl')

joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.pkl')

joblib.dump(bst,'model.pkl')


joblib.dump(vectorizer_keyword,'vectorizer_keyword.pkl')

joblib.dump(vectorizer_location,'vectorizer_location.pkl')

joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.pkl')

joblib.dump(bst,'model.pkl')