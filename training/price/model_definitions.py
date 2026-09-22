from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.price.feature_engineering import FEATURE_COLUMNS


NUMERIC_FEATURES = tuple(column for column in FEATURE_COLUMNS if column != "season")
CATEGORICAL_FEATURES = ("season", "series_id")


def build_preprocessor() -> ColumnTransformer:
    """series identity를 연속값으로 오인하지 않도록 명시적으로 one-hot 처리한다."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tabular_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
        ),
    }


def build_recurrent_model(
    model_type: str, input_shape: tuple[int, int], hidden_size: int = 32
):
    try:
        import tensorflow as tf
    except ImportError as exception:
        raise RuntimeError(
            "LSTM/GRU 비교를 실행하려면 TensorFlow가 필요합니다."
        ) from exception

    recurrent_layers = {
        "lstm": tf.keras.layers.LSTM,
        "gru": tf.keras.layers.GRU,
    }
    if model_type not in recurrent_layers:
        raise ValueError("지원하지 않는 순환 신경망 종류입니다: " + model_type)
    if hidden_size < 1:
        raise ValueError("순환 신경망 hidden size는 1 이상이어야 합니다.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            recurrent_layers[model_type](hidden_size),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name=f"price_{model_type}",
    )
    model.compile(optimizer="adam", loss="mse")
    return model
