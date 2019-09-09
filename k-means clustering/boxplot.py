import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart1.csv')
dataset1 = pd.read_csv('newk.csv')

#--------resting blood pressure before outlier detection
plt.figure(figsize=(7,7))
plt.title('resting blood pressure')
plt.boxplot(dataset.trestbps)
#--------resting blood pressure after outlier detection
plt.figure(figsize=(7,7))
plt.title('resting blood pressure')
plt.boxplot(dataset1.trestbps)
#--------cholestorol before outlier detection
plt.figure(figsize=(7,7))
plt.title('Cholestorol')
plt.boxplot(dataset.chol)
#--------cholestorol before outlier detection
plt.figure(figsize=(7,7))
plt.title('Cholestorol')
plt.boxplot(dataset1.chol)
#--------thalach before outlier detection
plt.figure(figsize=(7,7))
plt.title('thalach')
plt.boxplot(dataset.thalach)
#--------thalach after outlier detection
plt.figure(figsize=(7,7))
plt.title('thalach')
plt.boxplot(dataset1.thalach)

#--------oldpeak before outlier detection
plt.figure(figsize=(7,7))
plt.title('oldpeak')
plt.boxplot(dataset.oldpeak)
#--------oldpeak after outlier detection
plt.figure(figsize=(7,7))
plt.title('oldpeak')
plt.boxplot(dataset1.oldpeak)
#--------thal before outlier detection
plt.figure(figsize=(7,7))
plt.title('thal')
plt.boxplot(dataset.thal)
#--------thal after outlier detection
plt.figure(figsize=(7,7))
plt.title('thal')
plt.boxplot(dataset1.thal)