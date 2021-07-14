# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Debug Option
import pdb

# General Imports
import sys
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Imports SPN
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.algorithms.MPE import mpe

# Imports Bayessian Optimization
import GPyOpt
from GPyOpt.methods import BayesianOptimization

def rows_value_to_rows(rows_value):
  if rows_value == 0:
    rows = 'rdc'
  elif rows_value == 1:
    rows = 'kmeans'
  else:
    rows = 'gmm'
  return rows


def optimize_wine_bo_function(params):
  # Carga Dataset wine
  df_data = pd.read_pickle("wine_data.pkl")
  df_target = pd.read_pickle("wine_target.pkl")
  wine_data = df_data.to_numpy()
  wine_target = df_target.to_numpy().reshape(-1)

  f = open("wine_bo_hyperparams.txt", "a")

  # Hyperparams
  threshold, min_instances_slice, min_features_slice, rows_value = params[:,0], params[:,1], params[:,2], params[:,3]
  #rows
  rows = rows_value_to_rows(rows_value)

  f.write('Threshold=' + str(threshold) + ' - Min_instances_slice=' + str(min_instances_slice)  + ' - Min_features_slice=' + str(min_features_slice) + ' - Rows=' + str(rows))

  # K-Fold Cross Validation Params
  k = 2
  error = 0.0
  label = 13

  kf = KFold(n_splits=2)
  for train_index, test_index in kf.split(wine_data):
    x_train, x_test = wine_data[train_index], wine_data[test_index]
    y_train, y_test = wine_target[train_index], wine_target[test_index]

    # Prepare Train Data
    y_train_reshape = y_train.reshape(-1,1)
    train_data = np.hstack((x_train, y_train_reshape))
    
    # Learn SPN
    hyperparams = {"threshold": threshold, "min_instances_slice": min_instances_slice, "min_features_slice": min_features_slice, "rows": rows}
    try:
      spn_classification = learn_classifier(train_data,
                          Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Categorical]).add_domains(train_data),
                          learn_parametric, label, **hyperparams)

      # Prediction with MPE
      y_predict = np.empty(len(x_test))
      y_predict.fill(np.nan)
      y_predict_reshape = y_predict.reshape(-1,1)
      predict_data = np.hstack((x_test, y_predict_reshape))  
      predict_data = mpe(spn_classification, predict_data)

      # Calculate Error
      y_predict = predict_data[:,13]
      error += (1.0-accuracy_score(y_test, y_predict))
    except:
      error += 1.0
  error = error/float(k)

  f.write(' --> ERROR:' + str(error))

  if error == 1:
    f.write(' --> Fallo Prog: SI' + '\n')
  else:
    f.write(' --> Fallo Prog: NO' + '\n')

  f.close()
  #print(error)
  #print("ITERACION TERMINADA")

  return error

def main():
  seed = np.random.seed(int(sys.argv[1]))
  f = open("wine_bo_hyperparams.txt", "a")
  f.write('Seed ' + sys.argv[1] + '\n')
  f.close()
  # Hyperparams to optimize
  mixed_domain =[{'name': 'threshold', 'type': 'continuous', 'domain': (0,0.5),'dimensionality': 1}, # threshold of sifnificance
               {'name': 'min_instances_slice', 'type': 'discrete', 'domain': (0,100), 'dimensionality' : 1}, # minimum number of instances to slice
               {'name': 'min_features_slice', 'type': 'discrete', 'domain': (1, 3), 'dimensionality': 1}, # minimum number of features to slice
               {'name': 'rows', 'type': 'discrete', 'domain': (0,2),'dimensionality': 1}] # 0 -> rdc; 1 -> kmeans; 2 -> gmm
               

  # Bayesian Optimization Proccess
  bo_wine = BayesianOptimization(f=optimize_wine_bo_function,                     # Objective function       
                      domain=mixed_domain,          # Box-constraints of the problem
                      acquisition_type='EI',        # Expected Improvement
                      exact_feval = True)           # True evaluations, no sample noise
  
  # Number of Iterations for the Optimization
  max_iter = 30
  try:
    f = open("wine_bo_hyperparams.txt", "a")
    f.write('Run_Optimization \n')
    f.close()
    bo_wine.run_optimization(max_iter=max_iter, eps=1e-20)
    f = open("wine_bo_hyperparams.txt", "a")
    f.write('FIN EJECUCION \n')
    f.close()
  except:
    print("BUG.")

  # Print results
  result = "BO error --> " + str(bo_wine.fx_opt) + " with hyperparams: Threshold = " + str(bo_wine.x_opt[0]) + ", Min_instances_slice = " + str(bo_wine.x_opt[1]) + ", Min_features_slice = " + str(bo_wine.x_opt[2]) + ", Rows = " + rows_value_to_rows(bo_wine.x_opt[3]) + "."
  print(result)
  return result

if __name__ == "__main__":
    main()