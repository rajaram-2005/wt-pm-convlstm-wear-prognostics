# ConvLSTM Wear Prognostics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense

def build_convlstm(input_shape=(10, 64, 64, 1)):
    model = Sequential([
        ConvLSTM2D(32, kernel_size=(3,3), input_shape=input_shape, return_sequences=True, activation='relu'),
        BatchNormalization(),
        ConvLSTM2D(16, kernel_size=(3,3), return_sequences=False, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # RUL value
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
