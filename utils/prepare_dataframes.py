import pandas as pd
import numpy as np

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def load_data(path: str = "data/covid.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_data = pd.read_csv(path)
    df = full_data.copy(deep=True)

    return full_data, df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "entry_date",
        "date_symptoms",
        "date_died",
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            format="%d-%m-%Y",
            errors="coerce",
        )

    return df


def transform_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["died"] = df["date_died"].notna().astype(int)
    df["date_entry_symptom"] = (df["entry_date"] - df["date_symptoms"]).dt.days
    df["date_symptom_death"] = (df["date_died"] - df["date_symptoms"]).dt.days
    df = df[~(df["date_symptom_death"].fillna(0) < 0)]
    df = df.drop(columns="date_symptom_death")

    return df


def get_features(df: pd.DataFrame):
    features = [
        "sex",
        "age",
        "pneumonia",
        "pregnancy",
        "diabetes",
        "copd",
        "asthma",
        "inmsupr",
        "hypertension",
        "other_disease",
        "cardiovascular",
        "obesity",
        "renal_chronic",
        "tobacco",
        "contact_other_covid",
        "covid_res",
        "date_entry_symptom",
    ]

    target = "died"
    X = df[features]
    y = df[target]

    return X, y


def remove_leakage_features(X):
    leakage_features = [
        "icu",
        "intubed",
        "patient_type",
    ]
    X = X.drop(columns=leakage_features, errors="ignore")

    return X


def remove_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print("\nRemoving correlated features:")
    print(to_drop)

    X = X.drop(columns=to_drop)

    return X, to_drop


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_training_data(X_train, y_train):
    train_df = X_train.copy()
    train_df["died"] = y_train

    majority = train_df[train_df.died == 0]
    minority = train_df[train_df.died == 1]
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    balanced = pd.concat([majority, minority_upsampled])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X_bal = balanced.drop(columns="died")
    y_bal = balanced["died"]

    return X_bal, y_bal


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    ).reset_index(drop=True)

    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns).reset_index(
        drop=True
    )

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).reset_index(
        drop=True
    )

    return X_train, X_val, X_test


def build_dataframes(X_train, X_val, X_test, y_train, y_val, y_test):
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, val_df, test_df


def main():
    _, df = load_data()
    df = clean_dates(df)
    df = transform_date_features(df)
    X, y = get_features(df)
    X = remove_leakage_features(X)
    X, dropped = remove_correlated_features(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    X_train, y_train = balance_training_data(X_train, y_train)
    X_train, X_val, X_test = scale_features(
        X_train,
        X_val,
        X_test,
    )
    train_df, val_df, test_df = build_dataframes(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main()

    print("\nFinal dataset shapes:")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
