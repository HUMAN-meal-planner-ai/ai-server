import numpy as np
import tensorflow as tf

# 테스트용 시계열 데이터
X = np.random.rand(100, 30, 1).astype("float32")
y = np.random.rand(100, 1).astype("float32")

# LSTM
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 1)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

lstm_model.fit(X, y, epochs=1, batch_size=16, verbose=0)
print("LSTM OK:", lstm_model.predict(X[:3], verbose=0).flatten())

# GRU
gru_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 1)),
    tf.keras.layers.GRU(16),
    tf.keras.layers.Dense(1)
])

gru_model.compile(
    optimizer="adam",
    loss="mse"
)

gru_model.fit(X, y, epochs=1, batch_size=16, verbose=0)
print("GRU OK:", gru_model.predict(X[:3], verbose=0).flatten())