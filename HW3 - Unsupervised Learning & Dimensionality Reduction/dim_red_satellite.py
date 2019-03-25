import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import seaborn as sns

C = ['#ff400e',
'#f5892f',
'#20a8f6',
'#ebd250',
'#17f8b8',
'#1cc8dd',
'#6b0693',
'#307eea',
'#8dea7d',
'#5436b6',
'#8923a2']

sns.set(color_codes=True)

# df = pd.read_csv('./data_sets/wine.csv')
df = pd.read_csv('satellite.data', sep = ' ', header=None)      # 6435 instances
df_x = df.drop([36], axis=1)
df_y = df[[36]]

label = 36
# Train set
# df_y = df[label]
# df_x = df[[x for x in df.columns if label not in x]]

##########################################################################

nrange = [2,5,7,9,11,13,15]

df_x.dropna()

#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import random_projection
from sklearn.metrics import accuracy_score

#other imports 
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import validation_curve
from datetime import date
from sklearn.decomposition import KernelPCA

#Find the optimal component #
traindata = df
complist = [2,4,6,8,10,12,14,16]

for each in complist:
    comp = each

    # Fit the PCA analysis
    result = random_projection.GaussianRandomProjection(n_components=comp).fit(traindata)

    pca = sklearn.decomposition.PCA(n_components=comp)
    pca_transform = pca.fit_transform(df_x)
    var_values = pca.explained_variance_ratio_

    # print("Components"+ str(result.components_))
    print("Components "+ str(comp))

    explained_variance = np.var(pca_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    print("Explained Variance"+ str(explained_variance))
    # kpca_transform = kpca.fit_transform(df_x)
    # explained_variance = numpy.var(kpca_transform, axis=0)
    # explained_variance_ratio = explained_variance / numpy.sum(explained_variance)
    print("Explained Variance Ratio" + str(explained_variance_ratio))
    # print("RP Score"+ str(result.score(traindata, y= None)))
    # print("RP Score")
    # print pca.score(df)



print "PCA - kmeans"
for n in nrange:
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=28, n_init=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

print "PCA - EM"
for n in nrange:
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=28)
    em.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

reduced_data = PCA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=28, n_init=28)
kmeans.fit(reduced_data)

# # Graph
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig = plt.figure(1)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

print "paosdpfopasodpassssssssssssssssssssssss"
for index, row in df.iterrows():
    print index
    col = kmeans.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[2*col[0]], markersize=2)

plt.title('K-means clustering on the satellite dataset (PCA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('satellite_km_PCA.png')
plt.close(fig)

########################################################################

fig = plt.figure(2)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[2*int(row[label])], markersize=2)

plt.xticks(())
plt.yticks(())

plt.title('Each label mapped on the PCA-reduced 2D graph (satellite)')
fig.savefig('satellite_km_PCA_rankings.png')
plt.close(fig)

###########################################################################

reduced_data = PCA(n_components=2).fit_transform(df_x)
em = GaussianMixture(n_components=28)
em.fit(reduced_data)

# # Graph
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

fig = plt.figure(1)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    col = em.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=2)

plt.title('EM clustering on the satellite dataset (PCA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('satellite_em_PCA.png')
plt.close(fig)

##########################################################################

fig = plt.figure(2)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=2)

plt.xticks(())
plt.yticks(())

plt.title('Each label mapped on the PCA-reduced 2D graph (satellite)')
fig.savefig('satellite_em_PCA_rankings.png')
plt.close(fig)

#########################################################################

print "ICA - kmeans"
for n in nrange:

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = FastICA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=28, n_init=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

print "ICA - EM"
for n in nrange:

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = FastICA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=28)
    em.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

reduced_data = FastICA(n_components=2).fit_transform(df_x)
em = GaussianMixture(n_components=28)
em.fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 0.01, reduced_data[:, 0].max() + 0.01
y_min, y_max = reduced_data[:, 1].min() - 0.01, reduced_data[:, 1].max() + 0.01

fig = plt.figure(3)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    col = em.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=3)

plt.title('Expectation Maximization clustering on the satellite dataset (ICA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('satellite_em_ICA.png')
plt.close(fig)

# # ##########################################################################

fig = plt.figure(4)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=3)

plt.xticks(())
plt.yticks(())

plt.title('Labels mapped on the ICA-reduced 2D graph')
fig.savefig('satellite_em_ICA_rankings.png')

plt.close(fig)

# ##########################################################################

print "RP - kmeans"
for n in nrange:
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = KMeans(init='k-means++', n_clusters=28, n_init=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

print "RP - EM"
for n in nrange:
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = GaussianMixture(n_components=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

# ##########################################################################
df_x.dropna()
varss = [0.05, 0.28, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print "Feature Selection - kmeans"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = KMeans(init='k-means++', n_clusters=28, n_init=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

print "Feature Selection - EM"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = GaussianMixture(n_components=28)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(28):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == float(i):
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        if d: correct += max(d.values())
    
    print "Accuracy:", float(correct) / 6435

##########################################################################
