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
'#fc5819',
'#f87124',
'#f5892f',
'#f2a23a',
'#eeba45',
'#ebd250',
'#d1dd5e',
'#afe36d',
'#8dea7d',
'#6bf08c',
'#49f69b',
'#27fdaa',
'#17f8b8',
'#19e8c4',
'#1bd8d1',
'#1cc8dd',
'#1eb8ea',
'#20a8f6',
'#2496fb',
'#307eea',
'#3c66d9',
'#484ec7',
'#5436b6',
'#601ea4',
'#6b0693']

sns.set(color_codes=True)

# df = pd.read_csv('letter.csv')
# label = 'lettr'

df = pd.read_csv('satellite.data', sep = ' ', header=None)
traindata = df
df_x = df.drop([36], axis=1)
df_y = df[[36]]

# df = pd.read_csv('./spambase.data')
label = "satellite"

# Train set
# df_y = df[36]
# df_x = df[[x for x in df.columns if label not in x]]
# df_x = df[[x for x in df.columns if x != "satellite"]]


##########################################################################

reduced_data = PCA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
kmeans.fit(reduced_data)

df_x.dropna()


# Plot the decision boundary. For that, we will assign a color to each
# h = 1
# print reduced_data
# print len(reduced_data)
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# print "888888**************"
# print x_min
# print x_max
# print len(np.arange(x_min, x_max, h))
# print len(np.arange(y_min, y_max, h))
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# print xx
# print yy



# reduce num dimensions using PCA, then run kmeans clustering algo
print "PCA - kmeans"
for n in range (2,17,2):
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    # print reduced_data

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        # print "111============================="
        # print d
        # print "============================="
        # print d.values()
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
        
    
    print "Accuracy:", float(correct) / 4601

print "PCA - EM"
for n in range (2,17,2):
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=26)
    em.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
    
    print "Accuracy:", float(correct) / 4601


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
    



print '!!!!!!!!!!!!'
# # Graphical Representation of n = 3
# Fit the PCA analysis
result = PCA(n_components=3).fit(df)
from matplotlib.mlab import PCA as mlabPCA

mlab_pca = mlabPCA(df)

print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()













# # Graph
# # h = 1

# # # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# # print '!!!!!!!!!!!!2'
# # print len(xx)
# # print len(yy)
# # print len(xx.ravel())
# # print len(yy.ravel())
# # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# # Z = Z.reshape(xx.shape)
# # print '!!!!!!!!!!!!3'
# # fig = plt.figure(1)
# # plt.clf()

# # my_cmap = plt.cm.get_cmap('gist_ncar')
# # my_cmap.set_under('w')

# # plt.imshow(Z, interpolation='nearest',
# #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
# #            cmap=my_cmap,
# #            aspect='auto', origin='lower')

# # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# # centroids = kmeans.cluster_centers_
# # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)

# # plt.title('K-means clustering on the  dataset (PCA-reduced data)')

# # plt.xlim(x_min, x_max)
# # plt.ylim(y_min, y_max)
# # plt.xticks(())
# # plt.yticks(())
# # plt.show()
# # fig.savefig('satellite-letter_km_PCA.png')
# # plt.close(fig)

# ##########################################################################

fig = plt.figure(2)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[36])], markersize=2)

plt.xticks(())
plt.yticks(())

plt.title('Each label mapped on the PCA-reduced 2D graph')
fig.savefig('satellite-letter_km_PCA_letters.png')
plt.show()
plt.close(fig)

# ##########################################################################



print "ICA - kmeans"
# for n in range (2,17,2):

#     # print "Score:", pca.score(df)
#     print "N:", n

#     reduced_data = FastICA(n_components=n).fit_transform(df_x)
#     kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
#     kmeans.fit(reduced_data)

#     correct = 0
#     for i in range(26):
#         d = defaultdict(int)
#         for index, row in df.iterrows():
#             if row[36] == i:
#                 lab = kmeans.predict([reduced_data[index]])
#                 d[lab[0]] += 1
#         add_num_correct = max(d.values()) if len(d.values())>0 else 0
#         correct += add_num_correct
    
#     print "Accuracy:", float(correct) / 4601

print "ICA - EM"
for n in range (2,17,2):

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = FastICA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=26)
    em.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
    
    print "Accuracy:", float(correct) / 4601

reduced_data = FastICA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
kmeans.fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 0.01, reduced_data[:, 0].max() + 0.01
y_min, y_max = reduced_data[:, 1].min() - 0.01, reduced_data[:, 1].max() + 0.01

fig = plt.figure(3)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    col = kmeans.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=3)

plt.title('K-means clustering on the satellite dataset (ICA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('satellite-letter_km_ICA.png')
plt.close(fig)

# ##########################################################################

fig = plt.figure(4)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[36])], markersize=3)

plt.xticks(())
plt.yticks(())

plt.title('satellite - Each label mapped on the ICA-reduced 2D graph')
fig.savefig('satellite-letter_km_ICA_letters.png')

plt.close(fig)

##########################################################################



print "RP - kmeans"
for n in range (2,17,2):
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct

    
    print "Accuracy:", float(correct) / 4601

print "RP - EM"
for n in range (2,17,2):
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = GaussianMixture(n_components=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
    
    print "Accuracy:", float(correct) / 4601

##########################################################################

df_x.dropna()
varss = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print "Feature Selection - kmeans"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
    
    print "Accuracy:", float(correct) / 4601

print "Feature Selection - EM"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = GaussianMixture(n_components=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[36] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        add_num_correct = max(d.values()) if len(d.values())>0 else 0
        correct += add_num_correct
    
    print "Accuracy:", float(correct) / 4601

# ##########################################################################