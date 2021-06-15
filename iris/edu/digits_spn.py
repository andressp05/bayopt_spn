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


def main():
  seed = np.random.seed(int(sys.argv[1]))
  # Carga Dataset digits
  print('iteracion')
  df_data = pd.read_pickle("digits_data.pkl")
  df_target = pd.read_pickle("digits_target.pkl")
  digits_data = df_data.to_numpy()
  digits_target = df_target.to_numpy().reshape(-1)

  # K-Fold Cross Validation Params
  k = 2
  error = 0.0
  label = 64

  kf = KFold(n_splits=k)
  for train_index, test_index in kf.split(digits_data):
    X_train, X_test = digits_data[train_index], digits_data[test_index]
    y_train, y_test = digits_target[train_index], digits_target[test_index]

    # Prepare Train Data
    y_train_reshape = y_train.reshape(-1,1)
    train_data = np.hstack((X_train, y_train_reshape))

    # Learn SPN
    #hyperparams = {"cols": cols, "rows": rows, "threshold": threshold, "min_instances_slice": num_instances, "multivariate_leaf": False}
    #hyperparams = {"threshold": threshold}
    hyperparams = {"threshold": float(sys.argv[2]), "min_instances_slice": 60, "min_features_slice": 2, "rows": "kmeans", "cols" : "rdc"}
    try:
      spn_classification = learn_classifier(train_data,
                          Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, 
                                Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                                Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                                Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                                Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                                Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian,
                                Gaussian, Gaussian, Gaussian, Gaussian, Categorical]).add_domains(train_data),
                          learn_parametric, label, **hyperparams)

      # Prediction with MPE
      y_predict = np.empty(len(X_test))
      y_predict.fill(np.nan)
      y_predict_reshape = y_predict.reshape(-1,1)
      predict_data = np.hstack((X_test, y_predict_reshape))  
      predict_data = mpe(spn_classification, predict_data)

      # Calculate Error
      y_predict = predict_data[:,label]
      error += (1.0-accuracy_score(y_test, y_predict))
    except:
      error += 1.0
      print(' --> Fallo de Programacion: SI' + '\n')
    #print(error)
  error = error/float(k)

  print('ERROR: ' + str(error))

  return error

if __name__ == "__main__":
    main()
