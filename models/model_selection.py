from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def select_best_model(X, y):
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'LogisticRegression': LogisticRegression()
    }

    best_model, best_score = None, 0
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=5).mean()
        print(f'{name} Accuracy: {score:.4f}')
        if score > best_score:
            best_model, best_score = model, score
    
    return best_model

