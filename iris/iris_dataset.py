# Debug Option
import pdb

# General Imports
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

def main():
  iris = load_iris()
  idx = np.random.permutation(len(iris.data))
  iris_data,iris_target = iris.data[idx], iris.target[idx]
  df_data = pd.DataFrame(iris_data)
  df_target = pd.DataFrame(iris_target)
  df_data.to_pickle("iris_data.pkl")
  df_target.to_pickle("iris_target.pkl")
  
if __name__ == "__main__":
    main()