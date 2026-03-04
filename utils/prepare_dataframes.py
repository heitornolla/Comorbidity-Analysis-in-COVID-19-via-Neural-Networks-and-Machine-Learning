import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler



def load_data(path: str = 'data/covid.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_data = pd.read_csv(path)
    df = full_data.copy(deep=True)

    return full_data, df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        'entry_date', 
        'date_symptoms', 
        'date_died'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
    
    return df

def transform_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['died'] = df['date_died'].notna().astype(int) 
    
    df['date_entry_symptom'] = (df['entry_date'] - df['date_symptoms']).dt.days
    df['date_symptom_death'] = (df['date_died'] - df['date_symptoms']).dt.days

    df = df[~(df['date_symptom_death'].fillna(0) < 0)] 
    
    df = df.drop(columns='date_symptom_death')

    return df

def get_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        'sex', 'age', 'patient_type', 
        'icu', 'intubed', 'pneumonia', 
        'pregnancy', 'diabetes', 'copd', 
        'asthma', 'inmsupr', 'hypertension', 
        'other_disease', 'cardiovascular', 'obesity', 
        'renal_chronic', 'tobacco', 'contact_other_covid',
        'covid_res', 'date_entry_symptom'
    ]

    target = 'died'

    X = df[features]
    y = df[target]

    return X, y

def balance_dataset(X: pd.DataFrame, y: pd.DataFrame, target: str = 'died') -> pd.DataFrame:
    df_balanced = pd.concat([X, y], axis=1)

    df_survived = df_balanced[df_balanced[target] == 0]
    df_died = df_balanced[df_balanced[target] == 1]

    df_survived_downsampled = resample(
        df_survived,
        replace=False,
        n_samples=len(df_died),
        random_state=42
    )

    assert df_survived_downsampled.shape[0] == df_died.shape[0]

    return pd.concat([df_survived_downsampled, df_died])

def get_train_val_test_dfs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

    return train_df, val_df, test_df

def scale_features(train_df, val_df, test_df):
    scaler = StandardScaler()
    features = [c for c in train_df.columns if c != 'died']
    
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    return train_df, val_df, test_df


def main():
    _, df = load_data()
    df = clean_dates(df)
    df = transform_date_features(df)
    X, y = get_features(df)
    df = balance_dataset(X, y)
    train_df, val_df, test_df = get_train_val_test_dfs(df)
    train_df, val_df, test_df = scale_features(train_df, val_df, test_df)

    return train_df, val_df, test_df


if __name__ == '__main__':
    train_df, val_df, test_df = main()
    assert val_df.shape == test_df.shape
