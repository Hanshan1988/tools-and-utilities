# ecom-retention: pre-processing function
# imports
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

class CategoryImputer(BaseEstimator):

    def __init__(self, cols=[], thresholds=[], impute_with='other', return_df=False):
        # store parameters as public attributes
        self.thresholds = thresholds
        self.cols = cols
        self.return_df = return_df
        self.impute_with = impute_with
        
    def fit(self, X, y=None):
        # Assumes X is a DataFrame
        self._columns = X.columns.values
        # Create a dictionary mapping categorical column to unique values above threshold
        self._cat_cols = {}
        for col, thresh in zip(self.cols, self.thresholds):
            vc = X[col].value_counts()
            if thresh is not None:
                vc = vc[:thresh]
            vals = vc.index.values
            self._cat_cols[col] = vals
        return self
        
    def transform(self, X):
        # check that we have a DataFrame with same column names as 
        # the one we fit
        if not set(self._columns) <= set(X.columns):
            raise ValueError('Passed DataFrame does not contain the columns of fit DataFrame')

        # create separate array for new encoded categoricals
        X_cat = X.copy()
        for col, col_top_val in self._cat_cols.items():
            X_cat.loc[~X_cat[col].isin(col_top_val), col] = self.impute_with
                
        # return either a DataFrame or an array
        if self.return_df:
            return pd.DataFrame(data=X_cat, columns=self.cat_cols)
        else:
            return X_cat
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
class NumericConstantImputer(BaseEstimator):
    
    def __init__(self, cols=[], impute_vals=[], return_df=False):
        # store parameters as public attributes
        self.cols = cols
        self.impute_vals = impute_vals
        self.return_df = return_df
        
    def fit(self, X, y=None):
        self._columns = X.columns.values
        return self
        
    def transform(self, X):
        # check that we have a DataFrame with same column names as 
        # the one we fit
        if not set(self._columns) <= set(X.columns):
            raise ValueError('Passed DataFrame does not contain the columns of fit DataFrame')

        # create separate array for new encoded categoricals
        X_num = X.copy()
        # fill missing values
        for col, col_fill_val in zip(self.cols, self.impute_vals):
            if col_fill_val is not None:
                X_num[col] = X_num[col].fillna(col_fill_val)
                
        # return either a DataFrame or an array
        if self.return_df:
            return pd.DataFrame(data=X_num, columns=self.cols)
        else:
            return X_num
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def get_feature_names_mixed_types(column_transformer, cat_name='cat'):
    # @todo: assert column_transformer is an instance of sklearn's column transformer object
    col_name = []
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        if transformer_in_columns[0] == cat_name:
            names = transformer.get_feature_names() # ohe features
            for i in range(len(raw_col_name)):
                to_replace = f'x{i}_'
                replace_with = f'{raw_col_name[i]}||'
                names = [name.replace(to_replace, replace_with) for name in names]        
        else:
            names = raw_col_name
        col_name += names
    return col_name

# fit preprocessor
def preproc_fit(df=None, num_vars={}, catg_vars={}):
    assert df is not None, "Must pass dataframe containing relevant features"

    ## handle numeric variables
    numeric_features = list(num_vars.keys())
    numeric_imputes  = list(num_vars.values())
    
    # check that all columns passed are available
    num_miss = [k for k in numeric_features if k not in df.columns.tolist()]
    assert not len(num_miss), f"The following columns are missing from the dataframe: {num_miss}"
    numeric_transformer = Pipeline(steps=[('constant_imputer', NumericConstantImputer(numeric_features, numeric_imputes))])
    
    ## handle categorical variables
    categorical_features = list(catg_vars.keys())
    categorical_cutoffs  = list(catg_vars.values())
    
    # check that all columns passed are available
    catg_miss = [k for k in categorical_features if k not in df.columns.tolist()]
    assert not len(catg_miss), f"The following columns are missing from the dataframe: {catg_miss}"

    categorical_transformer = Pipeline(steps=[('cat_imputer', CategoryImputer(categorical_features, categorical_cutoffs)), 
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # transformed features    
    transformed_features = numeric_features + categorical_features
    passthrough_features = [col for col in df.columns if col not in transformed_features]
    preprocessor = ColumnTransformer(
    transformers=[
        ('passthrough', 'passthrough', passthrough_features),
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    df[categorical_features] = df[categorical_features].astype(str)
    df[numeric_features] = df[numeric_features].astype(float)

    # fit column transformer
    preprocessor.fit(df)
    
    return preprocessor


# apply preprocessor
def preproc_transform(df=None, preprocessor=None):
    
    # @todo - assert - on dataframes, instances
    assert df is not None, "Dataframe must not be none"
    
    # get features
    catg_features = preprocessor.transformers_[2][2]
    num_features  = preprocessor.transformers_[1][2]
    
    # assert that columns are available
    catg_miss = [k for k in catg_features if k not in df.columns]
    assert not len(catg_miss), f"Missing categorical variables from dataframe: {catg_miss}"
    num_miss  = [k for k in num_features if k not in df.columns]
    assert not len(num_miss), f"Missing numeric variables from dataframe: {num_miss}"
    
    # type conversion
    df[catg_features] = df[catg_features].astype(str)
    df[num_features]  = df[num_features].astype(float)
    
    # apply pre-processor
    X = preprocessor.transform(df)
    df_preproc = pd.DataFrame(X, columns=get_feature_names_mixed_types(preprocessor))
    print(f'=====> Completed transformation of features of shape {df_preproc.shape}')
    
    # convert to floats
    df_preproc = df_preproc.apply(lambda col: col.astype(float) if col.name.startswith('ft_') else col)

    return df_preproc
