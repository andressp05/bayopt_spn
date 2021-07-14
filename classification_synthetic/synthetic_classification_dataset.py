# Debug Option
import pdb

# General Imports
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
import pandas as pd

def main():
  classification_data, classification_target = make_classification(n_samples=1000, n_features=5, n_informative=4, n_redundant=1, n_classes=5)
  classification_data, classification_target = shuffle(classification_data, classification_target)
  df_data = pd.DataFrame(classification_data)
  df_target = pd.DataFrame(classification_target)
  df_data.to_pickle("classification_data.pkl")
  df_target.to_pickle("classification_target.pkl")
  
if __name__ == "__main__":
    main()