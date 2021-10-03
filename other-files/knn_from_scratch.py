import pandas as pd
import numpy as np
import math
import csv
from sklearn.preprocessing import StandardScaler
import random
import operator
import warnings
warnings.filterwarnings("ignore")

# Reading the dataset

data = pd.read_csv('galexa.csv')
# data.head()

# Independent Variables
X = data.iloc[:,1:]
# Dependent Variable (Lable Class)
Y = data.iloc[:,0]

# Feature Scaling (Essential as distance metric involved)

scaler = StandardScaler()
X = scaler.fit_transform(X)

data = pd.DataFrame(X)
data["class"] = Y

# data : Scaled Dataset
# data.head()


# Formatting the dataset as required for the KNN Algorithm

dataset = list()
with open("galexa_scaled.csv", 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)


# Calculating the euclidean distance between the two vectors/instances

def edistance(inst1, inst2):
  d = 0.0
  max_iter = len(inst1)-1
  i = 0
  while i < max_iter:
    d += (float(inst1[i]) - float(inst2[i]))**2
    i += 1
  return math.sqrt(d)


# Getting the nearest neighbors for the instance

def neighs(train_data, tst_data_row, no_of_neighs):
    distances = []

    max_iter = len(train_data)
    i = 0    
    while(i < max_iter):
        distance = edistance(tst_data_row,train_data[i],)
        distances.append((train_data[i],distance))
        i += 1

    distances = sorted(distances,key = lambda t:t[1])
    neighbors = []

    j = 0
    max_neigh = no_of_neighs
    while(j < max_neigh):
        neighbors.append(distances[j][0])
        j += 1

    return neighbors


# Predict the class

def class_predict(train_data, tst_data_row, no_of_neighs):
    neighbors = neighs(train_data, tst_data_row, no_of_neighs)
    output = []
    
    max_iter = len(neighbors)
    i = 0
    while i < max_iter:
        output.append(neighbors[i][-1])
        i += 1
    output_vals = set(output)
    
    prediction = max(output_vals, key=output.count)
    return prediction


# Accuracy Metric 

def calc_accuracy(y_true, y_pred):
  crt_count = 0
  max_iter = len(y_true)
  i = 0
  while i < max_iter:
    if y_true[i] != y_pred[i]:
      pass
    else:
      crt_count += 1
    i += 1
  result = crt_count / float(max_iter)
  return result


# KNN Algorithm

def knn_algo(train_data, test_data, no_of_neighs):
  p = list()
  max_iter = len(test_data)
  i = 0
  while i < max_iter:
    opt = class_predict(train_data,test_data[i], no_of_neighs)
    p.append(opt)
    i += 1
  
  return(p)


# KFold Validation : Splitting the dataset into k folds

def c_v_split(df, k_folds):
	df_split = list()
	copy_df = list(df)
	size = int(len(df) / k_folds)
	max_iter = k_folds
	i = 0
	while i < max_iter:
		f = list()
		while len(f) < size:
			index = random.randrange(len(copy_df))
			f.append(copy_df.pop(index))
		df_split.append(f)
		i += 1
	return df_split


# Finally evaluating the algorithm using a cross-validation-split

def evaluate(dataset, algo, k_folds, *args):
  folds = c_v_split(dataset, k_folds)
  res = list()
  max_iter = len(folds)
  i = 0
  while i < max_iter:
    train_s = list(folds)
    train_s.remove(folds[i])
    train_s = sum(train_s, [])
    test_s = list()
    fold = folds[i]
    for r in fold:
      copy_r = list(r)
      test_s.append(copy_r)
      copy_r[-1] = None
    y_pred = algo(train_s, test_s, *args)
    y_true = [r[-1] for r in fold]
    acc = calc_accuracy(y_true, y_pred)
    res.append(acc)
    i += 1

  return res

# Test Case 1

k_folds = 8
no_of_neighbors = 10
result = evaluate(dataset, knn_algo, k_folds, no_of_neighbors)

print('%s' % result)
print('Accuracy: %.3f%%' % (sum(result)/float(len(result))))