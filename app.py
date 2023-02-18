#Importing libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

# Loading the pickle files
vectorizer_keyword=joblib.load('./artifacts/vectorizer_keyword.pkl')
vectorizer_location=joblib.load('./artifacts/vectorizer_location.pkl')
tfidf_vectorizer=joblib.load('./artifacts/tfidf_vectorizer.pkl')
bst=joblib.load('./artifacts/model.pkl')

# Initiating Flask app
from flask import Flask,render_template,request,url_for
app = Flask(__name__)


#Home route
@app.route('/')
def hello_world():
    return 'Hello World!'

#Index route which directs the users to form page
@app.route('/index')
def form_input():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    
    #Query data
    location=request.form['location']
    keyword=request.form['keyword']
    message=request.form['message']
    
    #Using the pickle files we are transforming the query data
    location_transform=vectorizer_location.transform([location])
    keyword_transform=vectorizer_keyword.transform([keyword])
    message_transform=tfidf_vectorizer.transform([message])
  
    #Stacking everything together
    X_query=hstack([location_transform,keyword_transform,message_transform])
    X_query = xgb.DMatrix(X_query)
    #predicting on the query point
    pred=bst.predict(X_query)
    #considering 0.5 as thereshold
    if(pred>0.5):
        return 'It is a Disaster tweet'
    else:
        return 'The tweet is not related to diaster category'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)