import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer
from model import lstm

config = dict(
    input_len = 3,
    dim = 128,
    lstm_layers = 1,
    dense_layers = 2
)

def model_str(c):
    s = ""
    s += str(c["input_len"]) + "_"
    s += str(c["dim"]) + "_"
    s += str(c["lstm_layers"]) + "_"
    s += str(c["dense_layers"])
    return s

def train_model(epochs=1, load=False):
    tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")
    vocab = tokenizer.get_vocab()

    vocab_size = max(vocab.values()) + 1
    input_len = config["input_len"]
    dim = config["dim"]
    lstm_layers = config["lstm_layers"]
    dense_layers = config["dense_layers"]

    BATCH_SIZE = 2 ** 12

    print("Vocab size", vocab_size)

    if not load:
        model = lstm(vocab_size=vocab_size, input_len=input_len, dim=dim, lstm_layers=lstm_layers, dense_layers=dense_layers)
    else:
        model = tf.keras.models.load_model('./saved_models/' + model_str(config))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Create training data set
    print("Load data...")
    data = np.load("out.npy")#[:1000000]
    dlen = len(data)

    print("Done.")
    
    steps_per_epoch = dlen // BATCH_SIZE
    perm = np.random.permutation(steps_per_epoch) * BATCH_SIZE
    
    print("steps_per_epoch", steps_per_epoch)
    print("perm", perm[:10])
    
    def data_generator(dataset):
        #dataset = tf.constant(dataset)
        batch = 0
        while True:
            j = perm[batch]
            data_x = []
            data_y = [] 
            for b in range(BATCH_SIZE):
                i = j % (dlen - input_len - 1)
                train_x = dataset[i : i + input_len]
                train_y = dataset[i + input_len]
                #train_y = tf.keras.utils.to_categorical(train_y, num_classes=vocab_size)
                data_x.append(train_x)
                data_y.append(train_y)
                j += 1

            data_x = np.array(data_x)
            data_y = np.array(data_y)
            
            #print(data_x.shape)
            #print(data_y.shape)
            
            batch += 1
            batch = batch % steps_per_epoch

            yield (data_x, data_y)

    print("BATCHES IN TOTAL", steps_per_epoch)
    generator = data_generator(data)

    print("NEXT", next(generator))

    #print(next(generator))
    print("Fit the model...")
    # Fit the model
    tf.profiler.experimental.start("logdir")
    model.fit(x=generator, epochs=epochs, batch_size=BATCH_SIZE, use_multiprocessing=True, steps_per_epoch=steps_per_epoch)
    print("Done.")
    tf.profiler.experimental.stop()
    # Save the model
    model.save('./saved_models/' + model_str(config))
    print("Model saved.")
    return model

if __name__ == '__main__':
    model = train_model(epochs=15, load=True)

