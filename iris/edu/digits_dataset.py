# Debug Option
import pdb

# General Imports
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def main():
  digits = load_digits()
  idx = np.random.permutation(len(digits.data))
  digits_data, digits_target = shuffle(digits.data, digits.target)
  df_data = pd.DataFrame(digits_data)
  df_target = pd.DataFrame(digits_target)
  df_data.to_pickle("digits_data.pkl")
  df_target.to_pickle("digits_target.pkl")
  
if __name__ == "__main__":
    main()
