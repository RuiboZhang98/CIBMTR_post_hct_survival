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
    
    # This function uses a certein method to clean the data attribute
    # Updated (03/06): It does not directly alter the data attribute any more. 
    #                   Now it returns a view or a copy.
    def clean(self, method = "replace", params = 5):
        if method == "naive":
            return self.data.dropna()
        
        if method == "fdrop":
            # drops top {params} features with most missing values
            # Updated (03/06): It does not call dropna anymore
            assert isinstance(params,int), "in fdrop, params has to be a single integer"
            sorted_indices = np.argsort(self.data.count().to_numpy())
            return self.data.drop(self.data.columns[sorted_indices[:params]], axis = 1)
        
        if method == "rdrop":
            # drop the rows of data which have less than or equal to params number of missing values
            # Updated (03/06): Now it returns a view and it operates on data instead of raw_data
            assert isinstance(params,int), "in rdrop, params has to be a single integer"
            return self.data.dropna(thresh=self.data.keys().size-params)
                
        if method == "replace":
            # identify all entries that takes values in the list {params[0]} as missing values
            # params[1] present two possible options
            # "nan" means all the missing values are converted to np.nan
            # "missing" means catgorical missing values are converted to "missing"

            if isinstance(params, list):
                assert isinstance(params[0], list) & isinstance(params[1], str), \
                    r"The first element of list params has to be a list of column names. The second element must be one of 'nan' or 'missing'"
                target_values = params[0]
                mod = params[1]
            else:
                target_values = ["Not done", "Not tested", "Other", "Missing disease status", "Non-resident of the U.S."]
                mod = 'nan'
                print(f"by default the target values are {target_values}")
                print("These values are converted to np.nan")

            data_cleaned = self.data.copy(deep= True)
            cat_columns = self.data.select_dtypes(include = ['O']).columns
            if mod == 'nan':
                data_cleaned.loc[:,cat_columns] = data_cleaned[cat_columns].replace(target_values, np.nan)
                return data_cleaned
            
            if mod == 'missing':
                data_cleaned.loc[:,cat_columns] = data_cleaned[cat_columns].replace(target_values, 'missing')
                data_cleaned.loc[:,cat_columns] = data_cleaned[cat_columns].fillna('missing')
                num_columns = data_cleaned.select_dtypes(include = ['float64']).columns
                data_cleaned.loc[:, num_columns] = data_cleaned[num_columns].fillna(-1.0)
                return data_cleaned
            
            if mod == 'missing_cat':
                data_cleaned.loc[:,cat_columns] = data_cleaned[cat_columns].replace(target_values, 'missing')
                data_cleaned.loc[:,cat_columns] = data_cleaned[cat_columns].fillna('missing')
                return data_cleaned
            
    def report_missing_values(self, df, label1 = 'Feature', label2 = 'Percentage Missing'):
        # adapted from the implemention by Ray
        return pd.DataFrame(df.isna().sum()/df.shape[0] * 100).reset_index()\
        .rename(columns={"index":label1, 0:label2}).sort_values(by=label2, ascending=False)
    def impute(self, features, target):
        # split the data according whether the target feature is missing

        X = pd.get_dummies(self.data.dropna(subset=features)[features])

        y = self.data.loc[X.index][target]

        y_train = y.dropna().apply(str)

        X_train = X.loc[y_train.index]

        X_test  = X.loc[y.isna()]

        if (X_train.shape[0] > 0) and (X_test.shape[0] > 0):

            # train the logistic regression predictor 
    
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs') # For multiclass target
        
            model.fit(X_train, y_train)
     
            # impute the data
        
            y_pred = model.predict(X_test)
        
            if self.data.dtypes[target] == 'float64':
                self.data.loc[X_test.index, [target]] = list(map(float, y_pred))
            else:
                self.data.loc[X_test.index, [target]] = y_pred


