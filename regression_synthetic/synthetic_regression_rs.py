# Debug Option
import pdb

# General Imports
import sys
import math
from sklearn.metrics import mean_squared_error
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

def rows_value_to_rows(rows_value):
  if rows_value == 0:
    rows = 'rdc'
  elif rows_value == 1:
    rows = 'kmeans'
  else:
    rows = 'gmm'
  return rows


def optimize_regression_rs_function(threshold, min_instances_slice, min_features_slice, rows_value):
  # Carga Dataset Synthetic Regression
  df_data = pd.read_pickle("regression_data.pkl")
  df_target = pd.read_pickle("regression_target.pkl")
  regression_data = df_data.to_numpy()
  regression_target = df_target.to_numpy().reshape(-1)

  # Hyperparams
  threshold = threshold
  min_instances_slice = min_instances_slice
  min_features_slice = min_features_slice
  rows = rows_value_to_rows(rows_value)

  # K-Fold Cross Validation Params
  k = 2
  error = 0.0
  label = 30

  kf = KFold(n_splits=2)
  for train_index, test_index in kf.split(regression_data):
    #print("K-FOLD" + str(i))
    # Divide K-Fold Cross Validation
    x_train, x_test = regression_data[train_index], regression_data[test_index]
    y_train, y_test = regression_target[train_index], regression_target[test_index]

    # Prepare Train Data
    y_train_reshape = y_train.reshape(-1,1)
    train_data = np.hstack((x_train, y_train_reshape))
    
    # Learn SPN
    hyperparams = {"threshold": threshold, "min_instances_slice": min_instances_slice, "min_features_slice": min_features_slice, "rows": rows}
    try:
      spn_regression = learn_classifier(train_data,
                          Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian
                          , Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                          Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian]).add_domains(train_data),
                          learn_parametric, label, **hyperparams)

      # Prediction with MPE
      y_predict = np.empty(len(x_test))
      y_predict.fill(np.nan)
      y_predict_reshape = y_predict.reshape(-1,1)
      predict_data = np.hstack((x_test, y_predict_reshape))  
      predict_data = mpe(spn_regression, predict_data)

      # Calculate Error
      y_predict = predict_data[:,30]
      #print(y_test) 
      #print(y_predict.reshape(1,-1))
      error += mean_squared_error(y_test, y_predict)
    except:
      error += 999999.99
    #print(error)
  error = error/float(k)
  #print(error)
  #print("ITERACION TERMINADA")

  return error,threshold,min_instances_slice,min_features_slice,rows

def main():
  seed = np.random.seed(int(sys.argv[1]))
  
  f = open("regression_rs_hyperparams.txt", "w")
  f.write('Seed ' + sys.argv[1] + '\n')
  
  num_iterations = 30
  error_min = 999999.99

  for i in range(num_iterations):
    #print("Iteracion" + str(i) + '\n')
    # Hyperparams to optimize
    threshold = np.random.rand()/2.0
    min_instances_slice = np.random.randint(0,high=101)
    min_features_slice = np.random.randint(1, high=4)
    rows_value = np.random.randint(0,high=3)
    f.write(str(i) + ": threshold=" + str(threshold) + " - min_instances_slice=" + str(min_instances_slice) + " - min_features_slice=" + str(min_features_slice) + " - rows=" + str(rows_value))
  
    error,threshold,min_instances_slice,min_features_slice,rows = optimize_regression_rs_function(threshold, min_instances_slice, min_features_slice, rows_value)

    f.write(' --> ERROR:' + str(error))

    if error == 999999.99:
      f.write(' --> Fallo Prog: SI' + '\n')
    else:
      f.write(' --> Fallo Prog: NO' + '\n')

    if error < error_min:
      error_min = error
      threshold_min = threshold
      min_instances_slice_min = min_instances_slice
      min_features_slice_min = min_features_slice
      rows_min = rows
      

  #print("Value of (cols,rows,threshold,num_instances) that minimises the objective: ("+ cols_min + ", " + rows_min + ", " + str(threshold_min) + ", "+ str(num_instances_min)+ ")" ) 
  #print("Minimum error of the objective: ", error_min)
  f.close()
  result = "RS error --> " + str(error_min) + " with hyperparams: Threshold = " + str(threshold_min) + ", Min_instances_slice = " + str(min_instances_slice_min) + ", Min_features_slice = " + str(min_features_slice_min) + ", Rows = " + str(rows_min) + "."
  print(result)
  return result
  
if __name__ == "__main__":
    main()