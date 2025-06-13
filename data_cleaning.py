import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pdb


def load_and_prepare_df(path):
    df = pd.read_csv(path)
    # Rename target column
    df.rename(columns={"viewer_count": "current_viewers"}, inplace=True)
    # Convert timestamps
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
    # Remove "Special Events", "Just Chatting",  and "Summer Game Fest". 
    df = df[
        ~df["game_name"].isin(["Special Events", "Just Chatting", "Summer Game Fest"])
    ]
    # Remove rows with bad timestamps or missing viewers
    df = df.dropna(subset=["current_viewers", "started_at", "snapshot_time"])
    # Feature engineering: duration in minutes
    df["duration_min"] = (
        df["snapshot_time"] - df["started_at"]
    ).dt.total_seconds() / 60
    # Only keep positive durations and viewer counts
    df = df[(df["duration_min"] > 0) & (df["current_viewers"] > 0)]
    # limit the number of games we look at because there's over 9000.
    top_games = df["game_name"].value_counts().nlargest(280).index
    df["game_name"] = df["game_name"].where(df["game_name"].isin(top_games), "Other")
    # drop columns that we know won't can't be used in analysis
    drop_cols = [
        "id",
        "game_id",
        "type",
        "title",
        "user_id",
        "user_login",
        "user_name",
        "thumbnail_url",
        "snapshot_time",
        "started_at",
    ]
    df = df.drop(columns=drop_cols)
    return df.reset_index(drop=True)


def remove_ingest_canary(df):
    """Remove rows where user_login starts with 'ingest_canary'."""
    if "user_login" in df.columns:
        mask = ~df["user_login"].str.startswith("ingest_canary", na=False)
        df = df[mask].reset_index(drop=True)
    return df


def get_feature_pipeline(df):
    # Identify features
    numeric_features = ["duration_min"]
    categorical_features = ["language", "game_name", "is_mature"]
    # Pipelines
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def data_clean():
    df = load_and_prepare_df("data/twitch_streams_latest.csv")
    df = remove_ingest_canary(df)  # remove fake streams
    X = df.drop(columns=["current_viewers"])
    y = df["current_viewers"]
    # Split first (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Build pipeline
    preprocessor, numeric_features, categorical_features = get_feature_pipeline(df)
    full_pipeline = Pipeline([("preprocessor", preprocessor)])

    # Fit transform X_train, only transform X_test
    X_train_pre = full_pipeline.fit_transform(X_train)
    X_test_pre = full_pipeline.transform(X_test)
    # For reference: get feature names after encoding
    cat_feature_names = (
        full_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    feature_names = list(numeric_features) + list(cat_feature_names)
    print("Preprocessing complete. Shapes:")
    print("X_train_pre:", X_train_pre.shape)
    print("X_test_pre:", X_test_pre.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("Feature names:", feature_names[:10], "...")
    return (
        X_train_pre,
        X_test_pre,
        y_train,
        y_test,
        feature_names,
        full_pipeline,
        X_train,
        X_test,
    )


if __name__ == "__main__":
    data_clean()
