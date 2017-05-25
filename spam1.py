import pandas as pandas
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pandas.read_csv('SMSSpamCollection', delimiter='\t', header=None)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw) 
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression() 
classifier.fit(X_train, y_train) 
predictions = classifier.predict(X_test) 
for i, prediction in enumerate(predictions[:4]): 
    print ('Vorhersagen: %s. Nachricht: %s' % (prediction, X_test_raw[i]))

#print (df.head())
#print (df.count())

#print ("Anzahl den Spam-Nachrichten ", df[df[0] == 'spam'][0].count())
#print ('Anzahl den normalen Nachrichten:', df[df[0] == 'ham'][0].count())