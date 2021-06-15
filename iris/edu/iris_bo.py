# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Debug Option
import pdb

# General Imports
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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


def optimize_iris_bo_function(params):
  # Carga Dataset iris
  print('iteracion')
  df_data = pd.read_pickle("iris_data.pkl")
  df_target = pd.read_pickle("iris_target.pkl")
  iris_data = df_data.to_numpy()
  iris_target = df_target.to_numpy().reshape(-1)

  f = open("iris_bo_hyperparams.txt", "a")

  # Hyperparams
  #cols_value,rows_value,threshold,num_instances = params[:,0],params[:,1],params[:,2],int(params[:,3])
  threshold = params[:,0][0]
  #cols
  #cols = cols_value_to_cols(cols_value)
  ## A BORRAR cols = cols_value_to_cols(params['cols'][0])
  #rows
  #rows = rows_value_to_rows(rows_value)

  #f.write("cols=" + str(cols_value) + ' - rows=' + str(rows_value) + ' - threshold=' + str(threshold) + ' - num_instances=' + str(num_instances))
  f.write('threshold=' + str(threshold))

  # K-Fold Cross Validation Params
  k = 2
  error = 0.0

  kf = KFold(n_splits=2)
  for train_index, test_index in kf.split(iris_data):
    X_train, X_test = iris_data[train_index], iris_data[test_index]
    y_train, y_test = iris_target[train_index], iris_target[test_index]

    # Prepare Train Data
    y_train_reshape = y_train.reshape(-1,1)
    train_data = np.hstack((X_train, y_train_reshape))

    # Learn SPN
    #hyperparams = {"cols": cols, "rows": rows, "threshold": threshold, "min_instances_slice": num_instances, "multivariate_leaf": False}
    #hyperparams = {"threshold": threshold}
    hyperparams = {"threshold": threshold, "min_instances_slice": 1}
    try:
      spn_classification = learn_classifier(train_data,
                          Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Categorical]).add_domains(train_data),
                          learn_parametric, 4, **hyperparams)

      # Prediction with MPE
      y_predict = np.empty(len(X_test))
      y_predict.fill(np.nan)
      y_predict_reshape = y_predict.reshape(-1,1)
      predict_data = np.hstack((X_test, y_predict_reshape))  
      predict_data = mpe(spn_classification, predict_data)

      # Calculate Error
      y_predict = predict_data[:,4]
      error += (1.0-accuracy_score(y_test, y_predict))
    except:
      error += 1.0
    #print(error)
  error = error/float(k)

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
  '''
  mixed_domain =[{'name': 'cols', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # 0 -> rdc; 1 -> poisson
               {'name': 'rows', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # 0 -> rdc; 1 -> kmeans; 2 -> gmm
               {'name': 'threshold', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}, # threshold of sifnificance
               {'name': 'num_instances', 'type': 'continuous', 'domain': (0,300), 'dimensionality' : 1}] # minimum number of instances to split
               '''

  mixed_domain =[{'name': 'threshold', 'type': 'continuous', 'domain': (0,1),'dimensionality': 1}] # threshold of sifnificance

  bo_iris = BayesianOptimization(f=optimize_iris_bo_function,                     # Objective function       
                      domain=mixed_domain,          # Box-constraints of the problem
                      acquisition_type='EI',        # Expected Improvement
                      exact_feval = True)           # True evaluations, no sample noise
  
  # Number of Iterations for the Optimization
  max_iter = 2
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
    print("BUG.")
  result = "BO error --> " + str(bo_iris.fx_opt) + " with hyperparams: threshold = " + str(bo_iris.x_opt[0]) + "."
  print(result)
  return result

if __name__ == "__main__":
    main()
