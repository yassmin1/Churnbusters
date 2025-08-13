# data.py
"""
Data utilities for the IBM Telco Churn dataset.
- Loads raw CSV
- Splits into train/test
- Builds a ColumnTransformer (unfitted) that:
    * imputes & scales numeric features
    * imputes & one-hot encodes categoricals (drop='first', handle_unknown='ignore')
Returns raw X/y splits + the preprocessor (to be used inside an sklearn Pipeline).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


TELCO_URL = (
    "https://raw.githubusercontent.com/DiegoUsaiUK/"
    "Classification_Churn_with_Parsnip/master/00_Data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for Telco features."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # cleaner names
    )
    return preprocessor


def load_data(random_state: int = 42):
    """
    Load raw Telco data; return raw splits + unfitted preprocessor.
    Returns:
        X_train_raw, X_test_raw (pd.DataFrame)
        y_train, y_test (pd.Series)
        preprocessor (ColumnTransformer, unfitted)
        raw_feature_names (list[str]) - original feature columns (pre-transform)
    """
    df = pd.read_csv(TELCO_URL)

    # Clean numeric column that sometimes contains blanks
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target + features (keep *raw*; preprocessing happens in the Pipeline)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    # Train/test split done BEFORE fitting transformers (avoid leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(X)
    raw_feature_names = X.columns.tolist()

    return X_train_raw, X_test_raw, y_train, y_test, preprocessor, raw_feature_names
