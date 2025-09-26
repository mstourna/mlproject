from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_all_models(random_state=42):
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Support Vector Machine": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "AdaBoost": AdaBoostRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state),
        "LightGBM": LGBMRegressor(random_state=random_state)
    }

def get_model_params():
    return {
        "Linear Regression": {},
        "Ridge Regression": {'alpha': [0.1, 1.0, 10.0]},
        "Lasso Regression": {'alpha': [0.01, 0.1, 1.0]},
        "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        "Support Vector Machine": {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
        "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        "LightGBM": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]}
    }
