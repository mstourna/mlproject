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
from src.model_selector import get_all_models, get_model_params

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    expected_r2_score: float = 0.6
    overfitting_threshold: float = 0.1
    random_state: int = 42


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = model_trainer_config
        self.models = get_all_models(random_state=self.config.random_state)
        self.params = get_model_params()

    def load_data(self):
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        X_train = train_df.drop(columns=['math_score'], axis=1)
        y_train = train_df['math_score']
        X_test = test_df.drop(columns=['math_score'], axis=1)
        y_test = test_df['math_score']
        return X_train, y_train, X_test, y_test

    def load_preprocessor(self):
        with open(self.config.preprocessor_path, 'rb') as f:
            return pickle.load(f)

    def get_transformed_data(self, preprocessor, X_train, X_test):
        return preprocessor.transform(X_train), preprocessor.transform(X_test)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        return evaluate_model(X_train, y_train, X_test, y_test, self.models, self.params)

    def save_best_model(self, model):
        save_object(file_path=self.config.trained_model_file_path, obj=model)

    def initiate_model_trainer(self):
        logging.info("Starting model training pipeline")
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            preprocessor = self.load_preprocessor()
            X_train_transformed, X_test_transformed = self.get_transformed_data(preprocessor, X_train, X_test)

            model_report = self.train_and_evaluate(X_train_transformed, y_train, X_test_transformed, y_test)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = self.models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            if best_model_score < self.config.expected_r2_score:
                raise CustomException(f"No suitable model found with R2 above {self.config.expected_r2_score}", sys)

            # Overfitting/underfitting check
            train_pred = best_model.predict(X_train_transformed)
            test_pred = best_model.predict(X_test_transformed)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            if abs(train_r2 - test_r2) > self.config.overfitting_threshold:
                raise CustomException("Model may be overfitting or underfitting", sys)

            self.save_best_model(best_model)
            logging.info("Model training and saving complete.")
            return best_model_score

        except Exception as e:
            logging.error("Error in model training pipeline")
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
