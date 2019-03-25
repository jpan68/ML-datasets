import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture

raw_data = pd.read_csv('spambase.data', sep = ',', header=None)      # 4601 instances
char_x = raw_data.drop([57], axis=1)

raw_data = pd.read_csv('satellite.data', sep = ' ', header=None)      # 6435 instances
wine_x = raw_data.drop([36], axis=1)

# expectation maximization - elem can have a probability of being in a cluster
K = range(1, 40)
GMM_c = [GaussianMixture(n_components=k).fit(char_x) for k in K]
GMM_w = [GaussianMixture(n_components=k).fit(wine_x) for k in K]
print("Trained EM models")

LL_c = [gmm.score(char_x) for gmm in GMM_c]
LL_w = [gmm.score(wine_x) for gmm in GMM_w]
print("Calculated the log likelihood for each k")

BIC_c = [gmm.bic(char_x) for gmm in GMM_c]
BIC_w = [gmm.bic(wine_x) for gmm in GMM_w]
print("Calculated the BICs for each K")

AIC_c = [gmm.aic(char_x) for gmm in GMM_c]
AIC_w = [gmm.aic(wine_x) for gmm in GMM_w]
print("Calculated the AICs for each K")





plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, BIC_c, '*-', label='BIC for Spambase')
ax.plot(K, AIC_c, '*-', label='AIC for Spambase')
ax.plot(K, BIC_w, '*-', label='BIC for Satellite')
ax.plot(K, AIC_w, '*-', label='AIC for Satellite')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('Bayesian and Akaike Information Criterion Curve')
fig.savefig('graphs/em/bic_aic.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, LL_c, '*-', label='Spambase')
ax.plot(K, LL_w, '*-', label='Satellite')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve')
plt.legend(loc='best')
fig.savefig('graphs/em/log_likelihood.png')
plt.show()