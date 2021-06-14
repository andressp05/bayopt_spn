# Licensed under the BSD 3-clause license (see LICENSE.txt)

import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, MultivariateGaussian
from spn.structure.Base import Context

def main(job_id, params):
# Carga Dataset iris
  mixed_domain = {"cols": np.random.randint(0,high=2),
                "rows": np.random.randint(0,high=4),
                "threshold": np.random.rand(),
                "num_instances": np.random.randint(0, high=300)}
  #print(np.random.randint(0,high=2))
  #print(np.random.rand())
  iris = load_iris()
  idx = np.random.permutation(len(iris.data))
  print(idx)
  x,y = iris.data[idx], iris.target[idx]
  print(y)

  total_size = len(iris.data)
  fold_size = total_size // 5
  x_ndarray = iris.data
  x_lista = x_ndarray.tolist()
  x_test = x_lista[fold_size*2:fold_size*(2+1)]
  #print(type(x_test))
  #print(len(x_test))
  #print(x_test)
  y_ndarray = iris.target
  y_lista = y_ndarray.tolist()
  y_test = y_lista[fold_size*2:fold_size*(2+1)]
  x_train = []
  return
  y_train = []
  for j in range(5):
    if j == 2:
      continue
    x_train = x_train + x_lista[fold_size*j:fold_size*(j+1)]
    y_train = y_train + y_lista[fold_size*j:fold_size*(j+1)]
  x_train = np.asarray(x_train)
  x_test = np.asarray(x_test)
  print(x_test)
  print(len(x_test))
  y_train = np.asarray(y_train)
  print('ESTO')
  print(y_train)
  y_train_reshape = y_train.reshape(-1,1)
  print(y_train)
  train_data = np.hstack((x_train, y_train_reshape))
  print(train_data)
  to_predict = np.empty(len(x_test))
  to_predict.fill(np.nan)
  print(to_predict)
  y_predict_reshape = to_predict.reshape(-1,1)
  predict_data = np.hstack((x_test, y_predict_reshape))
  print(predict_data)
  y_predict = np.hsplit(predict_data, [4])
  print(y_predict)
  print(y_predict[1])
  #print(len(train_data))
  X = iris.data 
  y = iris.target.reshape(-1, 1)
  train_data_2 = np.hstack((X, y))
  #print(train_data_2)
  #print(len(train_data_2))
# CARGAR HIPERPARAMS --> setear
#                        --> entrenar
#                            --> evaluar
#  print(X_train)
#  print(X_test)
#  print(y_train)
#  print(y_test)
#  print(y_train_reshape)
  #sklearn_version = sklearn.__version__
  #print(sklearn_version)
  train_data = np.hstack((X_train, y_train_reshape))
  #print(train_data)
  spn_classification = learn_parametric(
            train_data,
            Context(
                parametric_types=[
                    MultivariateGaussian,
                    MultivariateGaussian,
                    MultivariateGaussian,
                    MultivariateGaussian,
                    Categorical,
                ]
            ).add_domains(train_data),
            multivariate_leaf=True,
            #ANDRES
        )
  print(spn_classification)   
  x = params['X'][0]
  print(x)
  y = params['Y'][0]
  z = params['prueba'][0]
  res = iris_spn(x, y, z)
  print('The Six hump camel back function:')
  print('\tf(%.4f, %0.4f, %.4f) = %f' % (x, y, z, res))
  return iris_spn(x, y, z)

if __name__ == "__main__":
    main(23, {'X': [0.0898], 'Y': [-0.7126]})