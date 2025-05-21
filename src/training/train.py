import argparse
from datetime import datetime
import joblib
import os
import pandas as pd
import yaml

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
# import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def read_table(filepath: str, parse_dates=None) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return pd.read_csv(filepath, parse_dates=parse_dates)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, parse_dates=parse_dates)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# TODO (Anusha/Shweta): Fill this function in with their code
def time_shift_data(X: pd.DataFrame, y: pd.DataFrame, horizon: int, context: int) -> list[pd.DataFrame, pd.DataFrame]:

    # print the original shapes
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # shift target price column by -forecast_horizon
    y_shifted = y.shift(-horizon)

    # drop rows with NaNs caused by shift (they occur at the end)
    valid_idx = y_shifted.dropna().index

    # align X and y so that input X[t] corresponds to output y[t+horizon]
    X_supervised = X.loc[valid_idx]
    y_supervised = y_shifted.loc[valid_idx]

    # print the shapes to confirm
    print(f"Shape of X_supervised: {X_supervised.shape}")
    print(f"Shape of y_supervised: {y_supervised.shape}")

    # Optional: Reset index if you want a clean DataFrame
    X_supervised = X_supervised.reset_index(drop=True)
    y_supervised = y_supervised.reset_index(drop=True)

    return X_supervised, y_supervised


def get_split_index(data_length: int, granularity: int, horizon: int) -> int:
    if granularity == 60:
        return data_length - horizon
    elif granularity == 30:
        return data_length - horizon * 2
    elif granularity == 15:
        return data_length - horizon * 4
    elif granularity == 5:
        return data_length - horizon * 12
    elif granularity == 120:
        return data_length - horizon // 2
    elif granularity == 180:
        return data_length - horizon // 3
    elif granularity == 240:
        return data_length - horizon // 4
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

def save_models(model_dict: dict, save_path: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = os.path.join(save_path, f"models_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # save each model using joblib
    for name, model in model_dict.items():
        save_path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"✅ Saved {name} model to {save_path}")

    return

def train_models(X, y, price_col, granularity=60, horizon=48, context=168, random_state=42, kwargs={}) -> None:
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
    forecast_horizon : int
        The number of time steps to make predictions for. Number of rows of test data.
    context_length : int
        The number of time steps of data to use as context for training. Number of rows of train data.
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

    # time shift the axes
    X_features, y = time_shift_data(X=X_features, y=y, horizon=horizon, context=context)

    # change this logic based on the data to make sure you forecast on 2 days
    split_idx = get_split_index(len(X_features), granularity=granularity, horizon=horizon)

    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define the model candidates that you want to use – you can change this
    model_constructors = {
        "XGBoost": XGBRegressor,
        # "LightGBM": LGBMRegressor,
        "CatBoost": CatBoostRegressor,
        "RandomForest": RandomForestRegressor,
        # "Ridge": Ridge,
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
        rmse = root_mean_squared_error(y_test, y_pred)

        models[model_name] = model
        metrics[model_name] = {"MAE": mae, "RMSE": rmse}

    print("\nTraining completed. Evaluation metrics:")
    for model_name, metric in metrics.items():
        print(f"{model_name}: MAE={metric['MAE']:.4f}, RMSE={metric['RMSE']:.4f}")

    save_path = kwargs.pop("save_path", "")
    save_models(models, save_path)

    return models, metrics


def main(args):
    # Load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print('config :: ', config)

    # Load feature and label data
    X = read_table(config["data_path"], parse_dates=[config["datetime_col"]])
    y_df = read_table(config["label_path"], parse_dates=[config["datetime_col"]])

    datetime_col = config["datetime_col"]
    assert datetime_col in X.columns and datetime_col in y_df.columns, \
        f"{datetime_col} must exist in both feature and label files"

    # Set datetime column as index for alignment --> take intersection
    X = X.set_index(datetime_col)
    y_df = y_df.set_index(datetime_col)
    common_index = X.index.intersection(y_df.index)

    # Filter X and y by overlapping range
    X = X.loc[common_index].sort_index()
    y_df = y_df.loc[common_index].sort_index()
    y = y_df[config["price_col"]]

    # Truncate to (context_length + forecast_horizon)
    total_length = len(X)
    horizon = config["forecast_horizon"]
    context = config["context_length"]

    if total_length < horizon + context:
        raise ValueError("Insufficient data: increase dataset size or reduce context/horizon.")

    X = X.iloc[-(context + horizon):].reset_index(drop=True)
    y = y.iloc[-(context + horizon):].reset_index(drop=True)
    print('Prepared X and y dataframes for training...')

    # Train models
    train_models(
        X=X,
        y=y,
        price_col=config["price_col"],
        granularity=config['granularity'],
        horizon=horizon,
        context=context,
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
    print('Parsed input arguments...')

    main(args)
