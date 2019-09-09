import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dataset = pd.read_csv('heart1.csv')
features = dataset.iloc[:,3:5].values
labels = dataset.iloc[:,13].values

#---------------Elbow method for finding cluster number
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()



clustering = KMeans(n_clusters = 4, init = 'k-means++',random_state=42)
#clustering.fit(features)

y_means = clustering.fit_predict(features)
y_means1 = list(y_means)
zero = y_means1.count(0)
one = y_means1.count(1)
two = y_means1.count(2)
three = y_means1.count(3)
#four = y_means1.count(4)
#five = y_means1.count(5)

#lowest_cluster_value = min(zero,one,two,three)
#print(lowest_cluster)
#lowest_cluster_number = 0
#if lowest_cluster_value == zero:
#    lowest_cluster_number = 0
#elif lowest_cluster_value == one:
#    lowest_cluster_number = 1
#elif lowest_cluster_value == two:
#    lowest_cluster_number = 2
#else:
#    lowest_cluster_number = 3
    
#print(lowest_cluster_number)    
q = 0
while q < len(y_means1):
    if y_means1[q] == 0:
        #print(q)
        dataset.drop(q,axis=0,inplace =True)
    q+=1

new = dataset.to_csv(r'C:\Users\muktadir\Desktop\final heart disease\kmeans\newk.csv',encoding='utf-8',index=False)
features_df = pd.DataFrame(features)
features_df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                      'exang','oldpeak','slope','ca','thal']
targets_df = pd.DataFrame(labels)
targets_df.columns = ['target']

color_theme = np.array(['darkgrey','powderblue','red','blue'])
plt.subplot(1,2,1)
plt.scatter(x = features_df.trestbps, y = features_df.chol, c = color_theme[clustering.labels_],s=50)
plt.title('k-means clustering')

#--------------- after outlier detection
dataset1 = pd.read_csv('newk.csv')
features = dataset1.iloc[:,:-1].values
labels = dataset1.iloc[:,13].values

clustering = KMeans(n_clusters = 3, random_state = 42)
kmeans = clustering.fit(features)

features_df1 = pd.DataFrame(features)
features_df1.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
                      'exang','oldpeak','slope','ca','thal']
targets_df1 = pd.DataFrame(labels)
targets_df1.columns = ['target']

color_theme1 = np.array(['darkgrey','powderblue','red'])
plt.subplot(1,2,2)
plt.scatter(x = features_df1.trestbps, y = features_df1.chol, c = color_theme[clustering.labels_],s=50)
plt.title('k-means clustering')