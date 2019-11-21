from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select provided features.

    Parameters
    ----------
    features : list(str)
        A list of feature names to select.
    """
    def __init__(self, features=None):
        self.features = features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Select columns from input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        """
        return X[self.features]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Prepare categorical features.

    Parameters
    ----------
    categories : ‘auto’ or a list of lists/arrays of values, default=’auto’.
        Categories (unique values) per feature:
            ‘auto’ : Determine categories automatically from the training data.
            list : categories[i] holds the categories expected in the ith
            column. The passed categories should not mix strings and numeric
            values within a single feature, and should be sorted in case of
            numeric values.
    """
    def __init__(self, categories='auto'):
        self.categories = categories
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Transform categorical data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input categorical features.
        """
        enc = OneHotEncoder(categories=self.categories, sparse=False)
        enc.fit(X)
        X_onehot = enc.transform(X)

        return X_onehot
