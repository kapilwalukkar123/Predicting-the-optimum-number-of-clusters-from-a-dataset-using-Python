# Predicting-the-optimum-number-of-clusters-from-a-dataset-using-Python



# Exploring unsupervised machine learning with the iris dataset

# importing all the required libraries to the python notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
%matplotlib inline

# Loading the iris dataset
iris =load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

# Displaying the whole dataset
iris_df

# Displaying the first 5 rows
iris_df.head()



# Finding the optimum number of clusters for k-means classification and also showing how to determine the value of K
x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph,
# allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()




# Applying k means to the dataset / Creating the k means classifier.
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()



# ** ** ** ** ** ** ** ** ** ** ** END ** ** ** ** ** ** ** ** ** ** ** ** **

