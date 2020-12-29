import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer
from model import lstm

def train_model(load=None):
    tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")
    vocab = tokenizer.get_vocab()

    vocab_size = max(vocab.values()) + 1
    input_len = 4
    dim = 64
    lstm_layers = 2

    BATCH_SIZE = 2 ** 12

    print("Vocab size", vocab_size)

    if load is None:
        model = lstm(vocab_size=vocab_size, input_len=input_len, dim=dim, lstm_layers=lstm_layers, dense_layers=2)
    else:
        model = tf.keras.models.load_model('./saved_models/' + load)
        model.vocab_size = max(vocab.values()) + 1
        model.input_len = 4
        model.dim = 64
        model.lstm_layers = 2
        model.desc = load

    print("MODEL:", model.desc)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #loss_weights=[1.0, 0.2],
    )

    # Create training data set
    print("Load data...")
    data = np.load("out.npy")
    dlen = len(data)

    print("Done.")

    def data_generator():
        j = 0
        while True:
            data_x = []
            data_y = []
            for b in range(BATCH_SIZE):
                i = j % (dlen - model.input_len - 1)
                train_x = data[i : i + model.input_len]
                train_y = data[i + model.input_len]
                train_y = tf.keras.utils.to_categorical(train_y, num_classes=model.vocab_size)
                data_x.append(train_x)
                data_y.append(train_y)
                j += 1

            data_x = np.array(data_x)
            data_y = np.array(data_y)

            yield (data_x, data_y)

    print("BATCHES IN TOTAL", dlen // BATCH_SIZE)
    generator = data_generator()

    print("NEXT", next(generator))

    #print(next(generator))
    print("Fit the model...")
    # Fit the model
    model.fit(x=generator, epochs=1, batch_size=BATCH_SIZE, use_multiprocessing=True)
    print("Done.")
    # Save the model
    model.save('./saved_models/' + model.desc)

    return model

if __name__ == '__main__':
    model = train_model()

