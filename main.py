from data.preprocess import preprocess_data
from models.model_selection import select_best_model
from models.hyperparameter_tuning import optimize_hyperparameters
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def automl_pipeline():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_train = preprocess_data(pd.DataFrame(X_train))
    X_test = preprocess_data(pd.DataFrame(X_test))
    
    # Select the best model
    best_model = select_best_model(X_train, y_train)
    
    # Optimize hyperparameters for the best model
    best_params = optimize_hyperparameters(best_model.__class__, X_train, y_train)
    best_model.set_params(**best_params)
    
    # Train the final model
    best_model.fit(X_train, y_train)
    accuracy = best_model.score(X_test, y_test)
    
    print(f'Final Model Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    automl_pipeline()

