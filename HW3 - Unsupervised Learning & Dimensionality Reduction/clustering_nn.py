import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


plt.style.use('ggplot')

NUM_CLUSTERS = 27
NUM_DIMENSIONS = 8
# df = pd.read_csv('./data_sets/spambase.data')
# df_x = df[[x for x in df.columns if x != 'spam']]

# df = pd.read_csv('spambase.data', sep = ',', header=None)
# df_x = df.drop([57], axis=1)
# label = 57

df = pd.read_csv('satellite.data', sep = ' ', header=None)
df_x = df.drop([36], axis=1)
df_y = df[[36]]

# df_y = df[[57]]


clusters_list = range(1, NUM_CLUSTERS, 2)
km, em, km_em = [], [], []
for clusters in clusters_list:
    print("CLUSTERS: {}".format(clusters))
    x = KMeans(n_clusters=clusters).fit_predict(df_x)
    df_x['kmeans'] = x
    train_x = df_x.iloc[0:4000]
    train_y = df[label].iloc[0:4000]
    test_x = df_x.iloc[4001:]
    test_y = df[label].iloc[4001:]
    clf = MLPClassifier(hidden_layer_sizes=(16 + 1)).fit(train_x, train_y)
    y = clf.predict(test_x)
    km.append(accuracy_score(test_y, y))
    print("KM: {}".format(km[-1]))
    df_x.drop('kmeans', axis=1, inplace=True)
    gau = GaussianMixture(n_components=clusters).fit(df_x)
    gx = gau.predict(df_x)
    df_x['em'] = gx
    train_x = df_x.iloc[0:4000]
    train_y = df[label].iloc[0:4000]
    test_x = df_x.iloc[4001:]
    test_y = df[label].iloc[4001:]
    clf = MLPClassifier(hidden_layer_sizes=(16 + 1)).fit(train_x, train_y)
    y = clf.predict(test_x)
    em.append(accuracy_score(test_y, y))
    print("EM: {}".format(em[-1]))
    df_x['kmeans'] = x
    train_x = df_x.iloc[0:4000]
    train_y = df[label].iloc[0:4000]
    test_x = df_x.iloc[4001:]
    test_y = df[label].iloc[4001:]
    clf = MLPClassifier(hidden_layer_sizes=(16 + 2)).fit(train_x, train_y)
    y = clf.predict(test_x)
    km_em.append(accuracy_score(test_y, y))
    print("KM + EM: {}\n".format(km_em[-1]))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(clusters_list, km, '*-', label='K-means')
ax.plot(clusters_list, em, '*-', label='EM')
ax.plot(clusters_list, km_em, '*-', label='K-means + EM')
ax.plot(clusters_list, [0.759] * len(clusters_list), label='Base accuracy')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Accuracy reached')
plt.legend(loc='best')
plt.title('Neural Net accuracy with clusters as attributes')
fig.savefig('graphs/clusters.png')
plt.show()