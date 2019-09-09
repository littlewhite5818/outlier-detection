import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
dataset = pd.read_csv('heart1.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,13].values

X_train, X_test, Y_train, Y_test = train_test_split(features,labels, test_size=0.25, random_state=0)
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]] You can use your data
plt.figure(figsize = (8,8))
print("\n===Gaussian===")
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, Y_train)
pred_gauss = gauss_clf.predict(X_test)
print("Predict Gaussian :",pred_gauss)
score_gauss_before = accuracy_score(Y_test, pred_gauss)
print("Accuracy Gaussian:", score_gauss_before)
probs = gauss_clf.predict_proba(X_test)
probs = probs[:, 1]
fpr, tpr, thresholds = precision_recall_curve(Y_test, probs)
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('recall')
plt.ylabel('precision')
# show the plot

plt.show()
plt.figure(figsize = (8,8))
print("\n===K Nearest Neighbors")
knneig = KNeighborsClassifier(n_neighbors=20)
knneig.fit(X_train, Y_train)
pred_knneigh = knneig.predict(X_test)
print("Predict Neighbors:",pred_knneigh)
score_knneigh_before = accuracy_score(Y_test, pred_knneigh)
print("Score KNeighnors :",score_knneigh_before)
#from sklearn.metrics import classification_report
#print(classification_report(Y_test, pred_knneigh))
probs = knneig.predict_proba(X_test)
probs = probs[:, 1]
fpr, tpr, thresholds = precision_recall_curve(Y_test, probs)
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('recall')
plt.ylabel('precision')
# show the plot

plt.show()

#
#---------------------- aFTER Outlier---------

dataset = pd.read_csv('newk.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,13].values

X_train, X_test, Y_train, Y_test = train_test_split(features,labels, test_size=0.25, random_state=0)
#X_test=[[48,1,2,110,229,0,0,168,0,1,3,0,7]] You can use your data

plt.figure(figsize = (8,8))
print("\n===Gaussian===")
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, Y_train)
pred_gauss = gauss_clf.predict(X_test)
print("Predict Gaussian :",pred_gauss)
score_gauss_after = accuracy_score(Y_test, pred_gauss)
print("Accuracy Gaussian:", score_gauss_after)
probs = gauss_clf.predict_proba(X_test)
probs = probs[:, 1]
fpr, tpr, thresholds = precision_recall_curve(Y_test, probs)
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('recall')
plt.ylabel('precision')
# show the plot

plt.show()


plt.figure(figsize = (8,8))
print("\n===K Nearest Neighbors")
knneig = KNeighborsClassifier(n_neighbors=20)
knneig.fit(X_train, Y_train)
pred_knneigh = knneig.predict(X_test)
print("Predict Neighbors:",pred_knneigh)
score_knneigh_after = accuracy_score(Y_test, pred_knneigh)
print("Score KNeighnors :",score_knneigh_after)
probs = knneig.predict_proba(X_test)
probs = probs[:, 1]
fpr, tpr, thresholds = precision_recall_curve(Y_test, probs)
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel('recall')
plt.ylabel('precision')
# show the plot

plt.show()

#from sklearn.metrics import classification_report
#print(classification_report(Y_test, pred_knneigh))


