from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def housing_stratified_shuffle_split(housing):
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing_strat_train = housing.loc[train_index]
        housing_strat_test = housing.loc[test_index]
    
    for set_ in (housing_strat_train, housing_strat_test):
        set_.drop("income_cat", axis=1, inplace=True) # DataFrame 객체의 drop() 메서드
    
    housing_train_X = housing_strat_train.drop("median_house_value", axis=1)
    housing_train_y = housing_strat_train["median_house_value"].copy()
    housing_test_X = housing_strat_test.drop("median_house_value", axis=1)
    housing_test_y = housing_strat_test["median_house_value"].copy()
    
    return housing_train_X, housing_train_y, housing_test_X, housing_test_y