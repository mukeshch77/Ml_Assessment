import numpy as np

def add_features(X):
    X = X.copy()

    # Log transform
    X["Amount_log"] = np.log1p(X["Amount"])

    # Square amount
    X["Amount_squared"] = X["Amount"] ** 2

    # Time based features
    X["Hour"] = (X["Time"] // 3600) % 24
    X["Is_night"] = X["Hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    return X
