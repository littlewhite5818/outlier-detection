from sklearn import ensemble
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv('heart.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,13].values

training_data = features[0:200]
test_data = features[200:]

#-----------naive bayes classifier
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.25,random_state = 0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb=gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
from sklearn.metrics import accuracy_score
nb_before_outlier = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred.round()))

#-----------Isolation Forest
our_anomaly_detector = ensemble.IsolationForest(contamination = 0.01)
our_anomaly_detector.fit(training_data)

training_predictions = our_anomaly_detector.predict(training_data)
testing_predictions = our_anomaly_detector.predict(test_data)
#print(training_predictions)
#print(testing_predictions)

#-------------Outlier Removal
q = 0
while q < len(training_predictions):
    if training_predictions[q] == -1:
        #print(q)
        dataset.drop(q,axis=0,inplace =True)
    q+=1
#print(q)    
q = 0
while q < len(testing_predictions):
    if testing_predictions[q] == -1:
        #print(q)
        dataset.drop(q+200,axis=0,inplace =True)
    q+=1
    

new = dataset.to_csv(r'C:\Users\muktadir\Desktop\pima-indians-diabetes-database\new1.csv',encoding='utf-8',index=False)

#------------Accuracy Evaluation of new dataset
dataset1 = pd.read_csv('new1.csv')


features = dataset1.iloc[:,:-1].values
labels = dataset1.iloc[:,13].values

#--------------------Naive Bayes Classifier
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.25,random_state = 0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb=gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
from sklearn.metrics import accuracy_score
nb_after_outlier = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred.round()))

plt.bar(1,nb_before_outlier,label ='naive bayes before outlier detection', width = 0.2)
plt.bar(1.1,nb_after_outlier,label ='naive bayes after outlier detection', width = 0.2)
#plt.bar(1.4,dt_before_outlier,label ='Naive bayes before outlier detection', width = 0.2)
#plt.bar(1.5,dt_after_outlier,label ='Naive bayes after outlier detection', width = 0.2)
plt.xticks([0.9,3.5])
plt.yticks([0.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
plt.xlabel('Models')
plt.ylabel('Acuracy')
plt.title('comparison')
plt.legend(loc='lower right')
plt.show()