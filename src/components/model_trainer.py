import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    expected_r2_score: float = 0.6
    overfitting_threshold: float = 0.1
    random_state: int = 42
    models: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
            "Random Forest": RandomForestRegressor(random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Support Vector Machine": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
            "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
            "XGBoost": XGBRegressor(random_state=self.random_state),
            "LightGBM": LGBMRegressor(random_state=self.random_state)
        }

        self.params = {
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

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self):
        logging.info("Entered the model trainer component")
        try:
            train_df = pd.read_csv(self.model_trainer_config.train_data_path)
            test_df = pd.read_csv(self.model_trainer_config.test_data_path)
            logging.info("Loaded training and testing data")

            X_train = train_df.drop(columns=['math_score'], axis=1)
            y_train = train_df['math_score']
            X_test = test_df.drop(columns=['math_score'], axis=1)
            y_test = test_df['math_score']

            logging.info("Split data into features and target")

            with open(self.model_trainer_config.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logging.info("Loaded preprocessor object")

            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Transformed training and testing data")

            model_report = evaluate_model(
                X_train_transformed, y_train,
                X_test_transformed, y_test,
                self.model_trainer_config.models,
                self.model_trainer_config.params
            )
            logging.info(f"Model evaluation report: {model_report}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = self.model_trainer_config.models[best_model_name]
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_r2_score:
                raise CustomException(f"No model found with R2 score above {self.model_trainer_config.expected_r2_score}", sys)

            train_pred = best_model.predict(X_train_transformed)
            test_pred = best_model.predict(X_test_transformed)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            logging.info(f"Train R2 score: {train_r2}, Test R2 score: {test_r2}")

            if abs(train_r2 - test_r2) > self.model_trainer_config.overfitting_threshold:
                raise CustomException("Model is overfitting or underfitting", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved best model at {self.model_trainer_config.trained_model_file_path}")

            return best_model_score

        except Exception as e:
            logging.error("Error occurred in model trainer")
            raise CustomException(e, sys)

# Test the trainer
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_csv, test_csv, preprocessor_pkl = transformation.initiate_data_transformation(train_path, test_path)

    print("Transformed train:", train_csv)
    print("Transformed test:", test_csv)
    print("Preprocessor saved at:", preprocessor_pkl)

    trainer = ModelTrainer()
    best_score = trainer.initiate_model_trainer()
    print("Best model R2 score:", best_score)
