"""Trains and evaluate models from a dataset"""

from create_dataframes import create_dataframes
from train_activity_models import train_activity_models
from train_ocupation_models import train_ocupation_models
from predictions import validate_models
from tune_hyperparameters_ocupation_models import parameter_search_all_ocupation_models
from tune_hyperparameters_activity_models import parameter_search_all_activity_models


def train_and_validate_models(path_file, tune_hyperparameters=False):
    """Train and evaluate models from data"""
    create_dataframes(path_file)
    if tune_hyperparameters:
        parameter_search_all_activity_models()
        parameter_search_all_ocupation_models()

    train_activity_models()
    train_ocupation_models()


if __name__ == "__main__":
    ORIGINAL_FILE = "Todo el dataset sin nan.xlsx"
    MOD_FILE = "Datos depurados a mano.xlsx"
    train_and_validate_models(MOD_FILE, tune_hyperparameters=False)
    validate_models()
