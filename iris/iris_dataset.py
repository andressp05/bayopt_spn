# Debug Option
import pdb

# General Imports
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd

def main():
  iris = load_iris()
  iris_data, iris_target = shuffle(iris.data, iris.target)
  df_data = pd.DataFrame(iris_data)
  df_target = pd.DataFrame(iris_target)
  df_data.to_pickle("iris_data.pkl")
  df_target.to_pickle("iris_target.pkl")
  
if __name__ == "__main__":
    main()