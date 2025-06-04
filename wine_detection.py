import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB,GaussianNB

wine=load_wine()
print(dir(wine))
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = wine.target
print(df.head())

mnb=MultinomialNB()
gnb=GaussianNB()
x=df.drop('target', axis=1)
y=df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=25)

mnb.fit(x_train, y_train)
print("MultinomialNB Score:", mnb.score(x_test, y_test))
gnb.fit(x_train, y_train)
print("GaussianNB Score:", gnb.score(x_test, y_test))
# Predicting the class of a new sample
new_sample = [[13.0, 2.5, 2.0, 18.0, 100.0, 2.5, 2.0, 0.3, 1.0, 3.0, 1.0, 2.5, 800.0]]
print("MultinomialNB Prediction:", mnb.predict(new_sample))
print("GaussianNB Prediction:", gnb.predict(new_sample))
# Saving the model using joblib