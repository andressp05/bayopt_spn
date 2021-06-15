# Debug Option
import pdb

# General Imports
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def main():
  wine = load_wine()
  idx = np.random.permutation(len(wine.data))
  #iris_data,iris_target = iris.data[idx], iris.target[idx]
  wine_data, wine_target = shuffle(wine.data, wine.target)
  df_data = pd.DataFrame(wine_data)
  df_target = pd.DataFrame(wine_target)
  df_data.to_pickle("wine_data.pkl")
  df_target.to_pickle("wine_target.pkl")
  
if __name__ == "__main__":
    main()
