import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from sklearn.naive_bayes import MultinomialNB

# your code here

#load the .env file variables
load_dotenv()
connection_string = os.getenv('DATABASE_URL')

df_raw = pd.read_csv(connection_string)

# remuevo la columna con el nombre del paquete

df_new = df_raw.drop('package_name', axis=1)

# cambio todos los comentarios a minuscula

df_new['review'] = df_new['review'].str.strip().str.lower()
df_new

# Divido el data set 
X = df_new['review']
y = df_new['polarity']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.25, random_state=50)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

vector_text = CountVectorizer(stop_words='english')
X_train = vector_text.fit_transform(X_train).toarray()
X_test = vector_text.transform(X_test).toarray()

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_train)

from sklearn import metrics

# Veamos la accuracy, precision y recall

print('Accuracy:',metrics.accuracy_score(y_test, model.predict(X_test)))
print('Precision:',metrics.precision_score(y_test, model.predict(X_test), average=None))
print('Recall:',metrics.recall_score(y_test, model.predict(X_test), average=None))

import pickle

Naive_Bayes = 'NaiveB_model.sav'
pickle.dump(model, open(Naive_Bayes,'wb'))

