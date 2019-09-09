import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('heart1.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,13].values

X_train, X_test, Y_train, Y_test = train_test_split(features,labels, test_size=0.25, random_state=0)
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]] You can use your data

print("\n===Gaussian===")
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, Y_train)
pred_gauss = gauss_clf.predict(X_test)
print("Predict Gaussian :",pred_gauss)
score_gauss_before = accuracy_score(Y_test, pred_gauss)
print("Accuracy Gaussian:", score_gauss_before)

print("\n===K Nearest Neighbors")
knneig = KNeighborsClassifier(n_neighbors=20)
knneig.fit(X_train, Y_train)
pred_knneigh = knneig.predict(X_test)
print("Predict Neighbors:",pred_knneigh)
score_knneigh_before = accuracy_score(Y_test, pred_knneigh)
print("Score KNeighnors :",score_knneigh_before)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred_knneigh))


#
#---------------------- aFTER Outlier---------

dataset = pd.read_csv('newk.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,13].values

X_train, X_test, Y_train, Y_test = train_test_split(features,labels, test_size=0.25, random_state=0)
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]] You can use your data


print("\n===Gaussian===")
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, Y_train)
pred_gauss = gauss_clf.predict(X_test)
print("Predict Gaussian :",pred_gauss)
score_gauss_after = accuracy_score(Y_test, pred_gauss)
print("Accuracy Gaussian:", score_gauss_after)


print("\n===K Nearest Neighbors")
knneig = KNeighborsClassifier(n_neighbors=20)
knneig.fit(X_train, Y_train)
pred_knneigh = knneig.predict(X_test)
print("Predict Neighbors:",pred_knneigh)
score_knneigh_after = accuracy_score(Y_test, pred_knneigh)
print("Score KNeighnors :",score_knneigh_after)
from sklearn.metrics import classification_report
print(classification_report(Y_test, pred_knneigh))


plt.figure(figsize=(8,8))

#plt.bar(1,score_tree_before,label ='naive bayes before outlier detection', width = 0.1)
#plt.bar(1.2,score_tree_after,label ='naive bayes after outlier detection', width = 0.1)
plt.bar(1,score_gauss_before,label ='naive bayes before outlier detection', width = 0.2)
plt.bar(1.2,score_gauss_after,label ='naive bayes after outlier detection', width = 0.2)
plt.bar(1.6,score_knneigh_before,label ='KNN before outlier detection', width = 0.2)
plt.bar(1.8,score_knneigh_after,label ='KNN after outlier detection', width = 0.2)
#plt.bar(2.2,score_randfor_before,label ='naive bayes before outlier detection', width = 0.1)
#plt.bar(2.4,score_randfor_after,label ='naive bayes after outlier detection', width = 0.1)
#plt.bar(1.4,dt_before_outlier,label ='Naive bayes before outlier detection', width = 0.2)
#plt.bar(1.5,dt_after_outlier,label ='Naive bayes after outlier detection', width = 0.2)
plt.xticks([0.9,3.5])
plt.yticks([0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
plt.xlabel('Models')
plt.ylabel('Acuracy')
plt.title('comparison')
plt.legend(loc='lower right')
plt.show()