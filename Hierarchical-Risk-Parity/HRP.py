import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# STEP 2: quasi-diagonalization
def get_quasi_diag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    # Get the first and the second item of the last tuple
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    # The total number of items is the third item of the last list
    num_items = link[-1, 3]
    # If the max of sort_ix is bigger than or equal to num_items
    while sort_ix.max() >= num_items:
        # Assign sort_ix index with 2 * num_items
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # Odd numbers as index
        df0 = sort_ix[sort_ix >= num_items]  # Find clusters
        # df0 contains even index and cluster index
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # Item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])  # Use pd.concat to combine Series
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

#STEP 3: recursive bisection
def get_cluster_var(cov, c_items):
    cov_ = cov.iloc[c_items, c_items] # matrix slice
    # calculate the inversev-variance portfolio
    ivp = 1./np.diag(cov_)
    ivp/=ivp.sum()
    w_ = ivp.reshape(-1,1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0,0]
    return c_var
def get_rec_bipart(cov, sort_ix):
    # compute HRP allocation
    # intialize weights of 1
    w = pd.Series(1, index=sort_ix)
    # intialize all items in one cluster
    c_items = [sort_ix]
    while len(c_items) > 0:
        # bisection
        """
        [[3, 6, 0, 9, 2, 4, 13], [5, 12, 8, 10, 7, 1, 11]]
        [[3, 6, 0], [9, 2, 4, 13], [5, 12, 8], [10, 7, 1, 11]]
        [[3], [6, 0], [9, 2], [4, 13], [5], [12, 8], [10, 7], [1, 11]]
        [[6], [0], [9], [2], [4], [13], [12], [8], [10], [7], [1], [11]]
        """
        c_items = [i[int(j):int(k)] for i in c_items for j,k in 
                   ((0,len(i)/2),(len(i)/2,len(i))) if len(i)>1]
        # now it has 2
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i] # cluster 1
            c_items1 = c_items[i+1] # cluter 2
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0/(c_var0+c_var1)
            w[c_items0] *= alpha
            w[c_items1] *=1-alpha
    return w

# Load your pricing data (replace 'pricing.csv' with your file)
assets = pd.read_csv('pricing.csv', index_col=0)
returns = assets.pct_change().dropna()
print(returns.head())

# STEP 1: hierarchical clustering
# Correlation matrix
corr = returns.corr()
# Distance matrix
d_corr = np.sqrt(0.5 * (1 - corr))
link = linkage(d_corr, 'single')
Z = pd.DataFrame(link)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()
print(Z)

res = get_quasi_diag(link)
print(res)

cov = returns.cov()
weights = get_rec_bipart(cov, res)
new_index = [returns.columns[i] for i in weights.index]
weights.index = new_index
print(weights)
