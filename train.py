from numpy import loadtxt, reshape, expand_dims
import numpy as np
import tflearn
from tflearn.layers.core import fully_connected, dropout, input_data
from tflearn.layers.estimator import regression
import pandas as pd
from sklearn import preprocessing
from  sklearn.model_selection import train_test_split
import urllib.request

# load the dataset
url = 'https://raw.githubusercontent.com/danlove99/Simple_Keras_Titanic/master/train.csv'
data = urllib.request.urlopen(url)
df = pd.read_csv(data)
y = df['Survived'].values
y.shape = (891, 1)
gender = pd.get_dummies(df['Sex'])
gender = gender['male']

# normalize fare and age data

fare = df['Fare']
normalized_fare = (fare - fare.min()) / (fare.max() - fare.min())
age = df['Age']
normalized_age = (age - age.min()) / (age.max() - age.min())
normalized_age = normalized_age.fillna(0.32000000)
parch = df['Parch']
pclass = df['Pclass']
sibsp = df['SibSp']

X = pd.concat([gender.reset_index(drop=True), normalized_fare.reset_index(drop=True),
               normalized_age.reset_index(drop=True), parch.reset_index(drop=True),
               pclass.reset_index(drop=True), sibsp.reset_index(drop=True)], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
x = expand_dims(X_train, axis=2)

nn = input_data(shape=[None, 6, 1], name='input')
nn = fully_connected(nn,16, activation='relu')
nn = dropout(nn, 0.8)
nn = fully_connected(nn,32, activation='relu')
nn = dropout(nn, 0.8)
nn = fully_connected(nn,64, activation='relu')
nn = dropout(nn, 0.8)
nn = fully_connected(nn,32, activation='relu')
nn = dropout(nn, 0.8)
nn = fully_connected(nn,16, activation='relu')
nn = dropout(nn, 0.8)
nn = fully_connected(nn, 1, activation='sigmoid')
nn = regression(nn, optimizer='adam', learning_rate=0.01, loss='binary_crossentropy', name='targets')
nn = tflearn.DNN(nn, tensorboard_dir='log')

nn.fit(x, y_train, n_epoch=150, snapshot_step=500, show_metric=True)
'''
_, accuracy = nn.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

nn.evaluate(X_test, y_test)'''