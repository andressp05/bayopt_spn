# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Debug Option
import pdb

# General Imports
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Imports SPN
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, MultivariateGaussian, Gaussian
from spn.structure.Base import Context
from spn.algorithms.MPE import mpe

# Imports Bayessian Optimization
import GPyOpt
from GPyOpt.methods import BayesianOptimization

# TO-DO si el tamaño no es divisible por k
def divide_k_folds(k, x, y, i):  
  # Calculate sizes
  total_size = len(x)
  fold_size = total_size // k

  # Convert ndarray to list
  x_list = x.tolist()
  y_list = y.tolist()
  
  # Create test dataset
  x_test_list = x_list[fold_size*i:fold_size*(i+1)]
  y_test_list = y_list[fold_size*i:fold_size*(i+1)]
  
  # Create train dataset
  x_train_list = []
  y_train_list = []
  for j in range(k):
    if j == i:
      continue
    x_train_list = x_train_list + x_list[fold_size*j:fold_size*(j+1)]
    y_train_list = y_train_list + y_list[fold_size*j:fold_size*(j+1)]

  # Convert back from list to ndarray
  x_test = np.asarray(x_test_list)
  y_test = np.asarray(y_test_list)
  x_train = np.asarray(x_train_list)
  y_train = np.asarray(y_train_list)

  return x_train, x_test, y_train, y_test

def cols_value_to_cols(cols_value):
  if cols_value == 0:
    cols = 'rdc'
  else:
    cols = 'poisson'
  return cols

def rows_value_to_rows(rows_value):
  if rows_value == 0:
    rows = 'rdc'
  elif rows_value == 1:
    rows = 'kmeans'
  else:
    rows = 'gmm'
  return rows


def optimize_iris_rs_function(cols_value, rows_value, threshold, num_instances):
  # Carga Dataset iris
  df_data = pd.read_pickle("iris_data.pkl")
  df_target = pd.read_pickle("iris_target.pkl")
  iris_data = df_data.to_numpy()
  iris_target = df_target.to_numpy().reshape(-1)

  # Hyperparams
  cols = cols_value_to_cols(cols_value)
  rows = rows_value_to_rows(rows_value)
  threshold = threshold
  num_instances = num_instances
  

  # K-Fold Cross Validation Params
  k = 5
  error = 0

  for i in range(k):
    #print("K-FOLD" + str(i))
    # Divide K-Fold Cross Validation
    x_train, x_test, y_train, y_test = divide_k_folds(k, iris_data, iris_target, i)
    #X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    # Prepare Train Data
    y_train_reshape = y_train.reshape(-1,1)
    train_data = np.hstack((x_train, y_train_reshape))
    
    # Learn SPN
    hyperparams = {"cols": cols, "rows": rows, "threshold": threshold, "min_instances_slice": num_instances, "multivariate_leaf": False}
    try:
      spn_classification = learn_classifier(train_data,
                          Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Categorical]).add_domains(train_data),
                          learn_parametric, 4, **hyperparams)

      # Prediction with MPE
      y_predict = np.empty(len(x_test))
      y_predict.fill(np.nan)
      y_predict_reshape = y_predict.reshape(-1,1)
      predict_data = np.hstack((x_test, y_predict_reshape))  
      predict_data = mpe(spn_classification, predict_data)

      # Calculate Error
      y_predict = np.hsplit(predict_data,[4])
      y_predict = y_predict[1]
      #print(y_test) 
      #print(y_predict.reshape(1,-1))
      error += (1-accuracy_score(y_test, y_predict))
    except:
      error += 1
    #print(error)

  error = error/k
  #print(error)
  #print("ITERACION TERMINADA")

  return error,cols,rows,threshold,num_instances

def main():
  seed = np.random.seed(int(sys.argv[1]))
  
  f = open("iris_rs_hyperparams.txt", "w")
  f.write('Seed ' + sys.argv[1] '\n')
  
  num_iterations = 100
  error_min = 100

  for i in range(num_iterations):
    
    # Hyperparams to optimize
    cols_value = np.random.randint(0,high=2)
    rows_value = np.random.randint(0,high=3)
    threshold = np.random.rand()
    num_instances = np.random.randint(0,high=300)
    f.write(str(i) + ": cols=" + str(cols_value) + ' - rows=' + str(rows_value) + ' - threshold=' + str(threshold) + ' - num_instances=' + str(num_instances))


    error,cols,rows,threshold,num_instances = optimize_iris_rs_function(cols_value,rows_value, threshold, num_instances)

    f.write(' --> ERROR:' + str(error))

    if error == 1:
      f.write(' --> Fallo de Programacion: SI' + '\n')
    else:
      f.write(' --> Fallo de Programacion: NO' + '\n')

    if error < error_min:
      error_min = error
      cols_min = cols
      rows_min = rows
      threshold_min = threshold
      num_instances_min = num_instances
    

  #print("Value of (cols,rows,threshold,num_instances) that minimises the objective: ("+ cols_min + ", " + rows_min + ", " + str(threshold_min) + ", "+ str(num_instances_min)+ ")" ) 
  #print("Minimum error of the objective: ", error_min)
  f.close()
  result = "RS error --> " + str(error_min) + " with hyperparams: cols = " + cols_min + ", rows = " + rows_min + ", threshold = " + str(threshold_min) + ", num_instances = " + str(int(num_instances_min)) + "."
  print(result)
  return result
  
if __name__ == "__main__":
    main()