def load_data(random_state=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import pandas as pd
    from sklearn.model_selection import train_test_split
    """
    Load IBM Telco Churn, clean, split, and preprocess with ColumnTransformer.
    Returns DataFrames for X_train/X_test (with final feature names),
    y_train/y_test, and the fitted preprocessor for reuse.
    """
    url = ("https://raw.githubusercontent.com/DiegoUsaiUK/"
           "Classification_Churn_with_Parsnip/master/00_Data/"
           "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Load
    df = pd.read_csv(url)
    df = df.dropna(subset=["TotalCharges"])  # drop rows with NaN in TotalCharges
    # Target & basic cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # may create NaNs → we will impute
    df.to_csv("data/cleaned_data.csv", index=False,)  # save cleaned data
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    # Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Split BEFORE fitting transformers (to avoid data leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Pipelines
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # keep original names where possible
    )

    # Fit on train only; transform both
    preprocessor.fit(X_train_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(preprocessor.transform(X_train_raw), columns=feature_names).astype("float64")
    X_test  = pd.DataFrame(preprocessor.transform(X_test_raw),  columns=feature_names).astype("float64")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

