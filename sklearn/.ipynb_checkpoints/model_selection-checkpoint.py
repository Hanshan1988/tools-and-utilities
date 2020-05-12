import pandas as pd
import numpy as np

def train_test_split_by_time(X, y, time_col, split_time, ts=None, drop_time_col=True):
    """
    Refer to sklearn's train_test_split.
    Input includes a time column name and a time cutoff to split on.
    """
    if time_col not in X.columns:
        ts.columns = [time_col]
        X = pd.concat([X, ts], axis=1)
    assert time_col in X.columns
    intime_idx = X.index[X[time_col] < split_time].tolist()
    outtime_idx = list(set(X.index) - set(intime_idx))
    X_train, X_test, y_train, y_test = X.iloc[intime_idx, :], X.iloc[outtime_idx, :], y.iloc[intime_idx, :], y.iloc[outtime_idx, :]
    X_train, X_test = X_train.drop(time_col, axis=1), X_test.drop(time_col, axis=1)
    return  X_train, X_test, y_train, y_test
