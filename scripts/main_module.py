import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

print("main module is loaded")
print(__file__)

class hct:
    def __init__(self, dir):
        self.dir = dir
        self.raw_data = pd.read_csv(self.dir)
        self.data = self.raw_data.copy(deep = True)

    def __str__(self):
        return f"raw data located at {self.dir}"
    
    def clean(self, method = "fdrop", params = 5):
        if method == "naive":
            self.data = self.raw_data.dropna()
        if method == "fdrop":
            assert isinstance(params,int), "in fdrop, params has to be a single integer"
            sorted_indices = np.argsort(self.raw_data.count().to_numpy())
            self.data = self.raw_data.drop(self.raw_data.columns[sorted_indices[:params]], axis = 1)
            self.data.dropna(inplace= True)
        if method == "rdrop":
            # drop the rows of data which have less than or equal to params number of missing values
            
            assert isinstance(params,int), "in rdrop, params has to be a single integer"
            self.data.dropna(thresh=self.raw_data.keys().size-params, inplace=True)
                
