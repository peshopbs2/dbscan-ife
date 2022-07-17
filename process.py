from operator import index
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import laplacian_kernel

path = pathlib.Path(__file__).parent.resolve()

def analyze(color, value):
    print(f'Diabetes cases in {color} cluster: ', end='')
    count = 0

    items = data[labels==value]
    glucose, bp, insulin, bmi = 0, 0, 0, 0

    for x in items:
        if(x[4] == 1):
            count+=1
            if x[0] > 126:
                glucose+=1
            
            if x[1] > 120:
                bp += 1
            
            if x[2] > 166:
                insulin += 1
            
            if x[3] > 25:
                bmi += 1

    print(f'Glucose: {glucose}, BP: {bp}, Insulin: {insulin}, BMI: {bmi}')

    print(f'diabetese: {count} / total: {items.size / 5}')
    if items.size == 0:
        return 0

    return count / items.size


# Importing the dataset
dataset = pd.read_csv(f'{path}/diabetes.csv')
data = dataset.iloc[:, [1, 2, 4, 5, 8]].values #Glucose, BP, Insulin, BMI, Output

maxcoef = 0
maxcolor = 0
properEps = 0

dbscan = DBSCAN(eps=25.5, min_samples=5)

labels = dbscan.fit_predict(data) 
cores = dbscan.core_sample_indices_.size
noise = len(data[labels == -1])
border = len(data) - cores - noise
print(f'cores = {cores}, border = {border}, noise = {noise}, all = {len(data)}')

#IFEs
mu = cores / (len(data))
ni = noise / (len(data))
pi = 1 - mu - ni

np.unique(labels)

plt.scatter(data[labels == -1, 2], data[labels == -1, 3], s = 10, c = 'black')
r = analyze('black', -1)

plt.scatter(data[labels == 0, 2], data[labels == 0, 3], s = 10, c = 'blue')
r = analyze('blue', 0)

plt.scatter(data[labels == 1, 2], data[labels == 1, 3], s = 10, c = 'red')
r = analyze('red', 1)

plt.scatter(data[labels == 2, 2], data[labels == 2, 3], s = 10, c = 'green')
r = analyze('green', 2)

plt.scatter(data[labels == 3, 2], data[labels == 3, 3], s = 10, c = 'brown')
r = analyze('brown', 3)

plt.xlabel('Insulin')
plt.ylabel('BMI')

plt.subplots_adjust(top=0.87, bottom=0.15)
plt.suptitle('DBSCAN over diabetes data', fontsize=10, fontweight='bold')
plt.figtext(0.05, 0.05, f"Degree of Membership: {mu:.3f}   Degree of Non-Membership: {ni:.3f}     Degree of Uncertainty: {pi:.3f}", fontsize = 9, fontweight='bold')

plt.show()
