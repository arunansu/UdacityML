
# coding: utf-8

# # Creating Customer Segments

# In this project you, will analyze a dataset containing annual spending amounts for internal structure, to understand the variation in the different types of customers that a wholesale distributor interacts with.
# 
# Instructions:
# 
# - Run each code block below by pressing **Shift+Enter**, making sure to implement any steps marked with a TODO.
# - Answer each question in the space provided by editing the blocks labeled "Answer:".
# - When you are done, submit the completed notebook (.ipynb) with all code blocks executed, as well as a .pdf version (File > Download as).

# In[1]:

# Import libraries: NumPy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tell iPython to include plots inline in the notebook
get_ipython().magic(u'matplotlib inline')

# Read dataset
data = pd.read_csv("wholesale-customers.csv")
print "Dataset has {} rows, {} columns".format(*data.shape)
print data.head()  # print the first 5 rows


# ##Feature Transformation

# **1)** In this section you will be using PCA and ICA to start to understand the structure of the data. Before doing any computations, what do you think will show up in your computations? List one or two ideas for what might show up as the first PCA dimensions, or what type of vectors will show up as ICA dimensions.

# Answer: First PCA dimension would be the line connecting points that have features close to mean. ICA dimensions will be the lines connecting modes.

# ###PCA

# In[3]:

# TODO: Apply PCA with the same number of dimensions as variables in the dataset
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data)

# Print the components and the amount of variance in the data contained in each dimension
print pca.components_
print pca.explained_variance_ratio_


# **2)** How quickly does the variance drop off by dimension? If you were to use PCA on this dataset, how many dimensions would you choose for your analysis? Why?

# Answer: The variance drops off rapidly after the first 2. I would choose 2 dimensions for PCA on this dataset.

# **3)** What do the dimensions seem to represent? How can you use this information?

# Answer: The dimensions represent composite features that we can use instead of all the features in the dataset to reduce list of features and still be able to use it for prediction.

# ###ICA

# In[4]:

# TODO: Fit an ICA model to the data
# Note: Adjust the data to have center at the origin first!
from sklearn.decomposition import FastICA
ica = FastICA()
ica.fit_transform(data)

# Print the independent components
print ica.components_


# **4)** For each vector in the ICA decomposition, write a sentence or two explaining what sort of object or property it corresponds to. What could these components be used for?

# Answer: These components represent a matrix that can be used to transform the given data projected on independent axis. 

# ##Clustering
# 
# In this section you will choose either K Means clustering or Gaussian Mixed Models clustering, which implements expectation-maximization. Then you will sample elements from the clusters to understand their significance.

# ###Choose a Cluster Type
# 
# **5)** What are the advantages of using K Means clustering or Gaussian Mixture Models?

# Answer: In K Means clustering we can pick the number of clusters we want and the algorithm will create exactly that many clusters. So each feature is assigned to exactly one cluster. But in Gaussian Mixture Model, each feature is assigned to one or many clusters with a certain probility. K Means clustering could be thought of as special case of Gaussian Mixed Model clustering, where the probability of a feature being in K clusters is 1 for a particular cluster and 0 for others.

# **6)** Below is some starter code to help you visualize some cluster data. The visualization is based on [this demo](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html) from the sklearn documentation.

# In[5]:

# Import clustering modules
from sklearn.cluster import KMeans
from sklearn.mixture import GMM


# In[14]:

# TODO: First we reduce the data to two dimensions using PCA to capture variation
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print reduced_data[:10]  # print upto 10 elements


# In[16]:

# TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
# The visualizer below assumes your clustering object is named 'clusters'

clusters = KMeans(init='k-means++', n_clusters=2, n_init=10)
clusters.fit(reduced_data)
print clusters


# In[17]:

# Plot the decision boundary by building a mesh grid to populate a graph.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
hx = (x_max-x_min)/1000.
hy = (y_max-y_min)/1000.
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

# Obtain labels for each point in mesh. Use last trained model.
Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])


# In[25]:

# TODO: Find the centroids for KMeans or the cluster means for GMM 
centroids = clusters.cluster_centers_
print centroids


# In[26]:

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# **7)** What are the central objects in each cluster? Describe them as customers.

# Answer: The central objects in each cluster represent a type of customer.

# ###Conclusions
# 
# ** 8)** Which of these techniques did you feel gave you the most insight into the data?

# Answer: PCA was helpful in identifying the transformed features that are useful to do clustering.

# **9)** How would you use that technique to help the company design new experiments?

# Answer: I would use clustering to create groups of items and people that are similar.

# **10)** How would you use that data to help you predict future customer needs?

# Answer: I would use the data to decide on what to stock in the store and what type of customers to cater to.
