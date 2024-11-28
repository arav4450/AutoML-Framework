import optuna
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning function that accepts the best model class
def optimize_hyperparameters(best_model_class, X, y):
    def objective(trial):
        # Define hyperparameter spaces based on model type
        if best_model_class.__name__ == 'RandomForestClassifier':
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            model = best_model_class(n_estimators=n_estimators, max_depth=max_depth)

        elif best_model_class.__name__ == 'SVC':
            C = trial.suggest_loguniform('C', 1e-3, 1e2)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            model = best_model_class(C=C, kernel=kernel)

        elif best_model_class.__name__ == 'LogisticRegression':
            C = trial.suggest_loguniform('C', 1e-3, 1e2)
            penalty = trial.suggest_categorical('penalty', ['l2', 'none'])
            model = best_model_class(C=C, penalty=penalty, solver='lbfgs', max_iter=1000)

        # Evaluate model using cross-validation
        return cross_val_score(model, X, y, cv=5).mean()

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print('Best Hyperparameters:', study.best_params)
    return study.best_params

