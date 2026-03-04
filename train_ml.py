import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from utils.eval_utils import evaluate_model


def run_ml_pipeline():
    from utils.prepare_dataframes import main as get_data
    
    train_df, val_df, test_df = get_data()
    
    X_train = train_df.drop('died', axis=1)
    y_train = train_df['died']
    X_val = val_df.drop('died', axis=1)
    y_val = val_df['died']
    X_test = test_df.drop('died', axis=1)
    y_test = test_df['died']

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # -1 train, 0 val
    split_indices = ([-1] * len(X_train)) + ([0] * len(X_val))
    pds = PredefinedSplit(test_fold=split_indices)

    models_config = {
        'SVM_Linear': {
            'model': SVC(kernel='linear', random_state=42),
            'params': {'C': [0.1, 1, 10, 100]}
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1, 10]}
        },
        "Gradient_Boosting": {
            "model": HistGradientBoostingClassifier(random_state=42),
            "params": {
                "max_iter": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [5, 11, 21]}
    },
        "Naive_Bayes": {
            "model": GaussianNB(),
            "params": {}
        }
    }

    for name, config in models_config.items():
        print(f'\n>>> Grid Search: {name}')

        grid = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=pds, 
            scoring='recall', 
            verbose=1
        )
        grid.fit(X_train_val, y_train_val)
        
        print(f'Best params {name}: {grid.best_params_}')

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
        evaluate_model(y_test, y_pred, model_name=f'{name}_Optimized')

if __name__ == '__main__':
    run_ml_pipeline()
