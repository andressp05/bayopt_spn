# Debug Option
import pdb

# General Imports
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
import pandas as pd

def main():
  digits = load_digits()
  digits_data, digits_target = shuffle(digits.data, digits.target)
  df_data = pd.DataFrame(digits_data)
  df_target = pd.DataFrame(digits_target)
  df_data.to_pickle("digits_data.pkl")
  df_target.to_pickle("digits_target.pkl")
  
if __name__ == "__main__":
    main()