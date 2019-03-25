import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv('spambase.data', sep = ',', header=None)
df_x = df.drop([57], axis=1)
df_y = df[[57]]

# df = pd.read_csv('satellite.data', sep = ' ', header=None)
# df_x = df.drop([36], axis=1)
# df_y = df[[36]]

# label = 57
label = 36

# df_y = df[label]
# df_x = df[[x for x in df.columns if label not in x]]

test_accuracy = []

hidden_layersizes = tuple(0 * [16])

##########################################################################

# Split into train and test
train_x = df_x.iloc[0:4000,:]
train_y = df_y.iloc[0:4000]
test_x = df_x.iloc[4001:,:]
test_y = df_y.iloc[4001:]

avg = []
for _ in range(0, 10):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layersizes)
    clf.fit(train_x, train_y)

    avg.append(accuracy_score(test_y, clf.predict(test_x)))
total = 0
for num in avg:
    total += num    
print 'Raw score avg', (total / len(avg))

##########################################################################

print "PCA"
for n in range(2,17,2):
    avg = []
    print "N:", n
    for _ in range(0, 10):
        pca = PCA(n_components=n)
        pca.fit(df_x)

        hidden_layersizes = tuple(13 * [10])

        # print "Score:", pca.score(df)
        

        reduced_data = PCA(n_components=n).fit_transform(df_x)

        # Split into train and test
        train_x = reduced_data[0:4000,:]
        train_y = df_y.iloc[0:4000]
        test_x = reduced_data[4001:,:]
        test_y = df_y.iloc[4001:]

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layersizes)
        clf.fit(train_x, train_y)

        avg.append(accuracy_score(test_y, clf.predict(test_x)))
    total = 0
    for num in avg:
        total += num
    print "Avg Score:", (total / len(avg))
    
##########################################################################

print "ICA"
for n in range(2,17,2):
    avg = []
    print "N:", n
    for _ in range(0, 10):
        # print "Score:", pca.score(df)
        

        hidden_layersizes = tuple(13 * [10])

        reduced_data = FastICA(n_components=n).fit_transform(df_x)

        # Split into train and test
        train_x = reduced_data[0:4000,:]
        train_y = df_y.iloc[0:4000]
        test_x = reduced_data[4001:,:]
        test_y = df_y.iloc[4001:]

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layersizes)
        clf.fit(train_x, train_y)

        avg.append(accuracy_score(test_y, clf.predict(test_x)))
    total = 0
    for num in avg:
        total += num
    print "Avg Score:", (total / len(avg))    

##########################################################################

print "RPs"
for n in range(2,17,2):
    avg = []
    print "N:", n
    for _ in range(0, 10):
        transformer = GaussianRandomProjection(n_components=n)
        reduced_data = transformer.fit_transform(df_x)

        hidden_layersizes = tuple(13 * [10])

        

        # Split into train and test
        train_x = reduced_data[0:4000,:]
        train_y = df_y.iloc[0:4000]
        test_x = reduced_data[4001:,:]
        test_y = df_y.iloc[4001:]

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layersizes)
        clf.fit(train_x, train_y)

        avg.append(accuracy_score(test_y, clf.predict(test_x)))
    total = 0
    for num in avg:
        total += num 
    print "Avg score:", (total / len(avg))        

##########################################################################