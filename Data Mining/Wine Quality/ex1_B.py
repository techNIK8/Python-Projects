#!/bin/bash
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from SVM_calc_fun import SVM_calc_fun
from sklearn.impute import KNNImputer

plt.style.use('fivethirtyeight')
filterwarnings('ignore')

df = pd.read_csv('winequality-red.csv')
# print(df.info())

pH_col = df['pH'].sample(frac=0.33, random_state=1111, axis=0)

# ---------------------------------------------------

# B1 Drop pH column
print("B1 Drop pH column")
dfB1 = df.drop(columns=['pH'])  # B1 11 c
# print(dfB1.info())
SVM_calc_fun(dfB1)

print(82 * '_')

# ---------------------------------------------------
# B2 mean value fill
print("B2 mean value fill")
dfB2 = df.drop(columns=['pH'])  # B1 11 c
pHcolB2 = df['pH'].drop(pH_col.index)
# print(pH_col.shape) # print oi times pou menoun
# print(df2)

mean = format(pH_col.mean(), '.2f')
# print(mean)

# fill in, empty values
newPhColB2 = pHcolB2.reindex(pd.RangeIndex(pHcolB2.index.max() + 3), fill_value=mean)
# print('newPhCol')
# print(newPhCol)

# insert newPhCol into dfB1 (dfB1 is without ph col)
dfB2.insert(8, 'pH', newPhColB2)

# convert object 'pH column' to float64
dfB2['pH'] = dfB2['pH'].astype(str).astype(float)
# change name of dataframe (not be similar to dfB1) & convert to df
dfB2 = pd.DataFrame(dfB2)
# print('DataFrame B2 column mean')
# print(dfB2.info())
# Show values of pH column
# print(dfB2['pH'])

# DFB2 to csv
# dfB2.to_csv('DFB2.csv')
SVM_calc_fun(dfB2)

print(82 * '_')

# ---------------------------------------------------
# B3 Fill empty values with Logistic Regression
print("B3 Fill empty values with Logistic Regression")
# create a pipeline object

X = df.drop(['quality'], axis=1)
y = df['quality']

# classifier = clf
clf = LogisticRegression(max_iter=7167, dual=False).fit(X, y)

fillNumB3 = float(format(clf.score(X, y), '.2f'))
# print(fillNumB3)

dfNoPhB3 = df.drop(columns=['pH'])

dfPhB3 = df['pH'].drop(pH_col.index)

# fill empty values of pH with fillNumB3
newPhColB3 = dfPhB3.reindex(pd.RangeIndex(dfPhB3.index.max() + 3), fill_value=fillNumB3)

# insert newPhCol into dfNoPh (dfNoPh is without ph col)
dfNoPhB3.insert(8, 'pH', newPhColB3)

# convert object 'pH column' to float64
dfNoPhB3['pH'] = dfNoPhB3['pH'].astype(str).astype(float)
# change name of dataframe (not be similar to dfB1) & convert to df
dfB3 = pd.DataFrame(dfNoPhB3)

# print('DataFrame B3 Logistic Regression')
# print(dfB3.info())
# print(newPhColB3)
# print(dfB3)

SVM_calc_fun(dfB3)

print(82 * '_')

# ---------------------------------------------------
# B4 K-means fill empty values
print("B4 K-means fill empty values")

# fit: Compute k-means clustering.
# max_iter = 1600/5=320 samples per centroid
# n_init = default 10 fores 8a trejei
# reduced_data = PCA(n_components=2).fit_transform(df)


kmeans = KMeans(n_clusters=5, max_iter=320, random_state=0).fit(df)  # Fitting the input data

# Getting the cluster labels
labels = kmeans.predict(df)
# print(labels)

# Centroid values
centroids = kmeans.cluster_centers_
# print(centroids)

# Predict the cluster for all the samples
P = kmeans.predict(df)

y_kmeans = kmeans.fit_predict(df)
# <class 'numpy.ndarray'>
# print(y_kmeans)

plt.figure(figsize=(6, 6))
plt.scatter(df.iloc[:, 8], df.iloc[:, 11])
plt.xlabel('Ph')
plt.ylabel('Quality')
plt.title('Visualization of raw data')
plt.show()

sse = []
list_k = list(range(1, 11))

# CHOOSING K
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Clusters')
plt.xlabel('Centroids')
plt.ylabel('Score')
plt.show()

dfNoPhB4 = df.drop(columns=['pH'])

dfPhB4 = df['pH'].drop(pH_col.index)
newPhColB4 = dfPhB4.reindex(pd.RangeIndex(dfPhB4.index.max() + 3), fill_value=np.NaN)

# insert newPhCol into dfNoPh (dfNoPh is without ph col)
dfNoPhB4.insert(8, 'pH', newPhColB4)

dfB4 = pd.DataFrame(dfNoPhB4)

imputer = KNNImputer(n_neighbors=5)
dfB4F = imputer.fit_transform(dfB4)
dfB4F = pd.DataFrame(dfB4F)

dfB4F.rename(columns={11: "quality"}, inplace=True)
SVM_calc_fun(dfB4F)

# print(dfB4.info())
