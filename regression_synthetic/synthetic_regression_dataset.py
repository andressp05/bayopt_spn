# Debug Option
import pdb

# General Imports
from sklearn.utils import shuffle
from sklearn.datasets import make_regression
import pandas as pd

def main():
  regression_data, regression_target = make_regression(n_samples=1000, n_features=30, n_informative=5)
  regression_data, regression_target = shuffle(regression_data, regression_target)
  df_data = pd.DataFrame(regression_data)
  df_target = pd.DataFrame(regression_target)
  df_data.to_pickle("regression_data.pkl")
  df_target.to_pickle("regression_target.pkl")
  
if __name__ == "__main__":
    main()