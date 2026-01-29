
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, RANDOM_STATE


def clean_data(df):
    """
    Removes duplicate rows from dataset
    """
    df = df.drop_duplicates()
    return df


def split_features_target(df):
    """
    Splits dataframe into features (X) and target (y)
    """
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y


def train_test_scaling(X, y, test_size=0.2):
    """
    Performs stratified train-test split.
    Scaling is handled inside ML pipeline.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test