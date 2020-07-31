import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from Kmeans import KMeans
import timeit
# from sklearn.cluster import KMeans

start = timeit.default_timer()

#Scikit - Learn
# km = KMeans()

# X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
# print(X.shape)
# y_pred = km.fit_predict(X)
# runs in 0.05sec

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)
    
clusters = len(np.unique(y))
print(clusters)
k = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = k.fit_predict(X)
k.plot()
# My from scratch algorithm runs in 2.26sec, If I were to optimize, I think I could get a very similar time
# as this is the first iteration of my algorithm.

stop = timeit.default_timer()

print('Time: ', stop - start)  
