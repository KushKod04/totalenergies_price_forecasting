import argparse
import os
import pandas as pd
import yaml

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error


def parse_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_table(filepath: str, parse_dates=None) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filepath, parse_dates=parse_dates)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, parse_dates=parse_dates)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def get_split_index(data_length: int, granularity: int) -> int:
    if granularity == 60:
        return data_length - 48
    elif granularity == 30:
        return data_length - 48 * 2
    elif granularity == 15:
        return data_length - 48 * 4
    elif granularity == 5:
        return data_length - 48 * 12
    elif granularity == 120:
        return data_length - 24
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

def train_models(X, y, price_col, granularity=60, random_state=42, kwargs={}):
    """
    Train multiple time series models given feature matrix X, target y, and price column name.

    Parameters:
    ----------
    X : pd.DataFrame
        Full feature matrix including all engineered features.
    y : pd.Series or np.array
        Target variable (price to predict).
    price_col : str
        Name of the price column in X (for reference).
    granularity : int
        The number of minutes between data samples.
    random_state : int
        Random state for reproducibility.

    Returns:
    -------
    models : dict
        Dictionary of trained models.
    metrics : dict
        Dictionary of evaluation metrics (MAE, RMSE) for each model.
    """

    models = {}
    metrics = {}

    # Optional: Drop price column from features if it's included
    feature_cols = [col for col in X.columns if col != price_col]
    X_features = X[feature_cols]

    # change this logic based on the data to make sure you forecast on 2 days
    split_idx = get_split_index(len(X), kwargs.pop("granularity"))

    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define the model candidates that you want to use – you can change this
    model_constructors = {
        "XGBoost": XGBRegressor,
        "LightGBM": LGBMRegressor,
        "CatBoost": CatBoostRegressor,
        "RandomForest": RandomForestRegressor,
        "Ridge": Ridge,
    }
    model_config = kwargs.get("models", {})

    # Train and evaluate each model
    for model_name, params in model_config.items():
        if model_name not in model_constructors:
            print(f"⚠️ Skipping unknown model: {model_name}")
            continue

        print(f"Training {model_name}...")

        if "random_state" in model_constructors[model_name]().__init__.__code__.co_varnames:
            params.setdefault("random_state", random_state)

        model = model_constructors[model_name](**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        models[model_name] = model
        metrics[model_name] = {"MAE": mae, "RMSE": rmse}

    print("\nTraining completed. Evaluation metrics:")
    for model_name, metric in metrics.items():
        print(f"{model_name}: MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}")

    return models, metrics


def main(args):
    # Load YAML
    config = parse_yaml_config(args.config)

    # Override YAML values from CLI if provided
    if args.forecast_horizon is not None:
        config["forecast_horizon"] = args.forecast_horizon
    if args.context_length is not None:
        config["context_length"] = args.context_length

    # Load feature and label data
    X = read_table(config["data_path"], parse_dates=[config["datetime_col"]])
    y_df = read_table(config["label_path"], parse_dates=[config["datetime_col"]])

    # Ensure alignment
    assert len(X) == len(y_df), "Mismatch between feature and label data lengths"

    y = y_df[config["price_column"]]

    # Truncate to (context_length + forecast_horizon)
    total_length = len(X)
    horizon = config["forecast_horizon"]
    context = config["context_length"]

    if total_length < horizon + context:
        raise ValueError("Insufficient data: increase dataset size or reduce context/horizon.")

    X = X.iloc[-(context + horizon):].reset_index(drop=True)
    y = y.iloc[-(context + horizon):].reset_index(drop=True)

    # Train models
    train_models(
        X=X,
        y=y,
        price_col=config["price_col"],
        granularity=config['granularity'],
        random_state=config["random_state"],
        kwargs=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time series models with config file and optional overrides.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    main(args)
