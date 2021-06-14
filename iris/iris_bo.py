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
  if cols_value < (1.0/2.0):
    cols = 'rdc'
  else:
    cols = 'poisson'
  return cols

def rows_value_to_rows(rows_value):
  if rows_value < (1.0/3.0):
    rows = 'rdc'
  elif rows_value < (2.0/3.0):
    rows = 'kmeans'
  else:
    rows = 'gmm'
  return rows


def optimize_iris_bo_function(params):
  # Carga Dataset iris
  df_data = pd.read_pickle("iris_data.pkl")
  df_target = pd.read_pickle("iris_target.pkl")
  iris_data = df_data.to_numpy()
  iris_target = df_target.to_numpy().reshape(-1)

  f = open("iris_bo_hyperparams.txt", "a")

  # Hyperparams
  cols_value,rows_value,threshold,num_instances = params[:,0],params[:,1],params[:,2],int(params[:,3])
  #cols
  cols = cols_value_to_cols(cols_value)
  ## A BORRAR cols = cols_value_to_cols(params['cols'][0])
  #rows
  rows = rows_value_to_rows(rows_value)

  f.write("cols=" + str(cols_value) + ' - rows=' + str(rows_value) + ' - threshold=' + str(threshold) + ' - num_instances=' + str(num_instances))

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

    # TO-DO --> donde estaba el smooth factor
    # A BORRAR Create SPN
    """ spn_classification = learn_parametric(
      train_data, 
      Context(
        parametric_types=[
          MultivariateGaussian,
          MultivariateGaussian,
          MultivariateGaussian,
          MultivariateGaussian,
          Categorical,
          ]).add_domains(train_data),
      cols = cols,
      rows = rows,
      threshold = threshold,
      min_instances_slice = num_instances
      ) """
    
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

  f.write(' --> ERROR:' + str(error))

  if error == 1:
    f.write(' --> Fallo de Programacion: SI' + '\n')
  else:
    f.write(' --> Fallo de Programacion: NO' + '\n')

  f.close()
  #print(error)
  #print("ITERACION TERMINADA")

  return error

def main():
  seed = np.random.seed(int(sys.argv[1]))
  f = open("iris_bo_hyperparams.txt", "a")
  f.write('Seed ' + sys.argv[1] + '\n')
  f.close()
  # Hyperparams to optimize
  mixed_domain =[{'name': 'cols', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # 0 -> rdc; 1 -> poisson
               {'name': 'rows', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # 0 -> rdc; 1 -> kmeans; 2 -> gmm
               {'name': 'threshold', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # threshold of sifnificance
               {'name': 'num_instances', 'type': 'continuous', 'domain': (0,300), 'dimensionality' : 1}] # minimum number of instances to split

  # Bayesian Optimization Proccess
  bo_iris = BayesianOptimization(f=optimize_iris_bo_function,                     # Objective function       
                      domain=mixed_domain,          # Box-constraints of the problem
                      acquisition_type='EI',        # Expected Improvement
                      exact_feval = True)           # True evaluations, no sample noise
  
  # Number of Iterations for the Optimization
  max_iter = 100
  try:
    #print("Entro Aquí")
    f = open("iris_bo_hyperparams.txt", "a")
    f.write('Run_Optimization \n')
    f.close()
    bo_iris.run_optimization(max_iter=max_iter, eps=1e-20)
    f = open("iris_bo_hyperparams.txt", "a")
    f.write('FIN EJECUCION \n')
    f.close()
    #print("OK")
  except:
    print("Error")
  # Print results
  result = "BO error --> " + str(bo_iris.fx_opt) + " with hyperparams: cols = " + cols_value_to_cols(bo_iris.x_opt[0]) + ", rows = " + rows_value_to_rows(bo_iris.x_opt[1]) + ", threshold = " + str(bo_iris.x_opt[2]) + ", num_instances = " + str(int(bo_iris.x_opt[3])) + "."
  print(result)
  return result

if __name__ == "__main__":
    main()