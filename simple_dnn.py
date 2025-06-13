from sklearn.metrics import mean_absolute_error,  r2_score
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
# Preprocess X_train, X_test with your pipeline, or use StandardScaler as needed
# X_train_pre = pipeline.fit_transform(X_train)
# X_test_pre = pipeline.transform(X_test_selected)
#TO-DO: try min-max scaler instead for preprocessing


def simp_dnn():
    # Build a simple neural network model
    model1 = keras.Sequential([
        keras.layers.Input(shape=(X_train_pre.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])

    model1.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Fit model
    history = model1.fit(X_train_pre, y_train_log, epochs=30, batch_size=64, validation_split=0.2)

    # Predict and inverse-transform
    y_pred_log = model1.predict(X_test_pre).flatten()
    y_pred = np.expm1(y_pred_log)
    print("MAE:", mean_absolute_error(y_test, y_pred))