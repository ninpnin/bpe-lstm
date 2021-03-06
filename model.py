import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer

def lstm(vocab_size=None, input_len=4, dim=64, lstm_layers=1, dense_layers=1):

    # Define model; inputs and the embedding
    inputs = tf.keras.Input(shape=(None,), name="input")
    x = tf.keras.layers.Embedding(vocab_size, dim)(inputs)
    
    # LSTM layers
    for i in range(lstm_layers - 1):
        x = tf.keras.layers.LSTM(dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(dim, return_sequences=False)(x)
    
    # Dense layers
    for i in range(lstm_layers - 1):
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
